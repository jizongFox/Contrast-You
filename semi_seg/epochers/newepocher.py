from typing import List

import numpy as np
import torch
from deepclustering2.configparser._utils import get_config  # noqa
from deepclustering2.meters2 import MeterInterface, AverageValueMeter
from deepclustering2.type import T_loss
from torch import Tensor

from contrastyou.losses.contrast_loss import SupConLoss3, is_normalized
from contrastyou.losses.iic_loss import _ntuple  # noqa
from contrastyou.projectors.nn import Normalize
from .comparable import InfoNCEEpocher
from .._utils import ContrastiveProjectorWrapper


@torch.no_grad()
def generate_similarity_masks(norm_feature1: Tensor, norm_feature2: Tensor):
    sim_matrix = norm_feature1.mm(norm_feature2.transpose(1, 0))  # ranging from -1 to 1
    sim_matrix -= -1
    sim_matrix /= 2
    return sim_matrix


class EncoderDenseContrastEpocher(InfoNCEEpocher):
    """
    adding a soften loss for both global and dense features.
    """

    def _init(self, *, reg_weight: float, projectors_wrapper: ContrastiveProjectorWrapper = None,
              infoNCE_criterion: T_loss = None, **kwargs):
        super()._init(reg_weight=reg_weight, projectors_wrapper=projectors_wrapper, infoNCE_criterion=infoNCE_criterion,
                      **kwargs)
        self._infonce_criterion2 = SupConLoss3()

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(EncoderDenseContrastEpocher, self)._configure_meters(meters)
        config = get_config(scope="base")
        # todo: adding try except within a context manager
        for f in config["ProjectorParams"]["GlobalParams"]["feature_names"] or []:
            m = "global"
            meters.register_meter(f"{f}_{m}/origin_mi", AverageValueMeter(), )
            meters.register_meter(f"{f}_{m}/weight_mi", AverageValueMeter(), )
        for f in config["ProjectorParams"]["DenseParams"]["feature_names"] or []:
            m = "dense"
            meters.register_meter(f"{f}_{m}/origin_mi", AverageValueMeter(), )
            meters.register_meter(f"{f}_{m}/weight_mi", AverageValueMeter(), )

        return meters

    def _global_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                        label_group) -> Tensor:

        # global loss with hard coded loss.
        unregulated_loss = super()._global_infonce(feature_name=feature_name, proj_tf_feature=proj_tf_feature,
                                                   proj_feature_tf=proj_feature_tf, partition_group=partition_group,
                                                   label_group=label_group)

        def soft_global_infonce(soft_criterion: SupConLoss3):
            sim_coef = generate_similarity_masks(proj_tf_feature, proj_feature_tf)
            regularized_loss = soft_criterion(proj_tf_feature, proj_feature_tf,
                                              pos_weight=sim_coef)
            return regularized_loss

        regularized_loss = soft_global_infonce(self._infonce_criterion2)

        self.meters[f"{feature_name}_global/origin_mi"].add(unregulated_loss.item())
        self.meters[f"{feature_name}_global/weight_mi"].add(regularized_loss.item())

        config = get_config(scope="base")
        reg_weight = float(config["ProjectorParams"]["GlobalParams"]["softweight"])

        return unregulated_loss + reg_weight * regularized_loss

    def _dense_infonce_for_encoder(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                                   label_group):
        # get unregularized loss based on index of the dense feature map.
        unregularized_loss = super()._dense_infonce_for_encoder(feature_name=feature_name,
                                                                proj_tf_feature=proj_tf_feature,
                                                                proj_feature_tf=proj_feature_tf,
                                                                partition_group=partition_group,
                                                                label_group=label_group)
        # repeat a bit
        assert "Conv" in feature_name, feature_name

        proj_tf_feature, proj_feature_tf = self._reshape_dense_feature(proj_tf_feature, proj_feature_tf)

        b, hw, c = proj_feature_tf.shape

        if not (is_normalized(proj_feature_tf, dim=2) and is_normalized(proj_feature_tf, dim=2)):
            proj_feature_tf = Normalize(dim=2)(proj_feature_tf)
            proj_tf_feature = Normalize(dim=2)(proj_tf_feature)
        proj_feature_tf, proj_tf_feature = proj_feature_tf.reshape(-1, c), proj_tf_feature.reshape(-1, c)
        sim_coef = generate_similarity_masks(proj_feature_tf, proj_tf_feature)
        regularized_loss = self._infonce_criterion2(proj_tf_feature, proj_feature_tf,
                                                    pos_weight=sim_coef)
        self.meters[f"{feature_name}_dense/origin_mi"].add(unregularized_loss.item())
        self.meters[f"{feature_name}_dense/weight_mi"].add(regularized_loss.item())

        config = get_config(scope="base")
        reg_weight = float(config["ProjectorParams"]["DenseParams"]["softweight"])

        return unregularized_loss + reg_weight * regularized_loss

    def _dense_infonce_for_decoder(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                                   label_group):
        raise RuntimeError(f"{self.__class__.__name__} does not support contrasting on dense feature from decoder")


class EncoderDenseMixupContrastEpocher(InfoNCEEpocher):
    """
    This epocher inherits the infoNCE epocher and then apply mixup for it.
    """

    def _init(self, *, reg_weight: float, projectors_wrapper: ContrastiveProjectorWrapper = None,
              infoNCE_criterion: T_loss = None, **kwargs):
        super()._init(reg_weight=reg_weight, projectors_wrapper=projectors_wrapper, infoNCE_criterion=infoNCE_criterion,
                      **kwargs)
        self._infonce_criterion2 = SupConLoss3()

    def regularization(self, *, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int,
                       label_group: List[str], partition_group: List[str], unlabeled_image=None, **kwargs):
        reg_loss = super(EncoderDenseMixupContrastEpocher, self).regularization(
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf,
            seed=seed, label_group=label_group, partition_group=partition_group
        )
        # creating mixup images
        b, c, h, w = unlabeled_image.shape
        rand_index = np.random.permutation(list(range(b)))
        rand_index_ = rand_index[::-1].copy()
        mu = torch.rand(b, 1, 1, 1, device=self.device)
        assert torch.logical_and(mu >= 0, mu <= 1).any()
        mixed_image = mu * unlabeled_image[torch.from_numpy(rand_index)] + \
                      (1 - mu) * unlabeled_image[torch.from_numpy(rand_index_)]

        assert torch.logical_and(mixed_image >= 0, mixed_image <= 1).all()  # make sure that

        mixed_logits = self._model(mixed_image)

        return reg_loss

    def _dense_infonce_for_decoder(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                                   label_group):
        raise RuntimeError(f"{self.__class__.__name__} does not support contrasting on dense feature from decoder")
