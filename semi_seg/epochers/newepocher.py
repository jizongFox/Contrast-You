from typing import List, Any

import numpy as np
import torch
from deepclustering2.configparser._utils import get_config  # noqa
from deepclustering2.meters2 import MeterInterface, AverageValueMeter
from deepclustering2.type import T_loss
from deepclustering2.utils import simplex, class2one_hot, one_hot
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

from contrastyou.featextractor import FeatureExtractorWithIndex as FeatureExtractor
from contrastyou.helper import weighted_average_iter
from contrastyou.losses.contrast_loss import SupConLoss3, is_normalized, SupConLoss4
from contrastyou.losses.iic_loss import _ntuple  # noqa
from contrastyou.projectors.nn import Normalize
from . import unl_extractor, ProjectionHead
from .comparable import InfoNCEEpocher
from .._utils import ContrastiveProjectorWrapper

__variable_dict = {}


def register_variable(*, name: str, object_: Any):
    __variable_dict[name] = object_


def get_variable(*, name: str):
    return __variable_dict[name]


@torch.no_grad()
def generate_similarity_masks(norm_feature1: Tensor, norm_feature2: Tensor):
    sim_matrix = norm_feature1.mm(norm_feature2.transpose(1, 0))  # ranging from -1 to 1
    sim_matrix -= -1
    sim_matrix /= 2
    return sim_matrix


@torch.no_grad()
def generate_similarity_weightmatrix_based_on_labels(label_dist1, label_dist2):
    assert simplex(label_dist1, axis=1) and simplex(label_dist2, axis=1)
    _norm_class = Normalize(dim=1)
    norm_dist1, norm_dist2 = _norm_class(label_dist1), _norm_class(label_dist2)
    sim_distance = norm_dist1.mm(norm_dist2.t())
    assert torch.logical_and(sim_distance >= 0, sim_distance <= 1).any()
    return sim_distance


@torch.no_grad()
def generate_mixup_image(image_list1: Tensor, image_list2: Tensor, label_list1: Tensor, label_list2: Tensor):
    b, c, *hw = image_list1.shape
    assert one_hot(label_list1) and one_hot(label_list2)
    mu = torch.rand(b, 1, 1, 1, device=image_list1.device)
    mixed_image = image_list1 * mu + image_list2 * (1 - mu)
    assert torch.logical_and(mixed_image >= 0, mixed_image <= 1).all()  # make sure that
    mixed_target = mu.squeeze()[..., None] * label_list1 + (1 - mu.squeeze()[..., None]) * label_list2
    assert simplex(mixed_target), mixed_target
    return mixed_image, mixed_target


class EncoderDenseContrastEpocher(InfoNCEEpocher):
    """
    adding a soften loss for both global and dense features.
    """

    def _init(self, *, reg_weight: float, projectors_wrapper: ContrastiveProjectorWrapper = None,
              infoNCE_criterion: T_loss = None, **kwargs):
        super()._init(reg_weight=reg_weight, projectors_wrapper=projectors_wrapper, infoNCE_criterion=infoNCE_criterion,
                      **kwargs)
        self._infonce_criterion2 = SupConLoss3()  # adding a soften loss

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(EncoderDenseContrastEpocher, self)._configure_meters(meters)
        config = get_config(scope="base")
        # todo: adding try except within a context manager
        for f in config["ProjectorParams"]["GlobalParams"]["feature_names"] or []:
            m = "global"
            meters.register_meter(f"{f}_{m}/hardcode_mi", AverageValueMeter(), )
            meters.register_meter(f"{f}_{m}/soften_mi", AverageValueMeter(), )
        for f in config["ProjectorParams"]["DenseParams"]["feature_names"] or []:
            m = "dense"
            meters.register_meter(f"{f}_{m}/hardcore_mi", AverageValueMeter(), )
            meters.register_meter(f"{f}_{m}/soften_mi", AverageValueMeter(), )

        return meters

    def _global_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                        label_group) -> Tensor:

        # global loss with hard coded loss.
        unregulated_loss = super(EncoderDenseContrastEpocher, self)._global_infonce(
            feature_name=feature_name,
            proj_tf_feature=proj_tf_feature,
            proj_feature_tf=proj_feature_tf,
            partition_group=partition_group,
            label_group=label_group
        )

        def soft_global_infonce(soft_criterion: SupConLoss3):
            sim_coef = generate_similarity_masks(proj_tf_feature, proj_feature_tf)
            regularized_loss = soft_criterion(proj_tf_feature, proj_feature_tf,
                                              pos_weight=sim_coef)
            return regularized_loss

        regularized_loss = soft_global_infonce(self._infonce_criterion2)

        self.meters[f"{feature_name}_global/hardcode_mi"].add(unregulated_loss.item())
        self.meters[f"{feature_name}_global/soften_mi"].add(regularized_loss.item())

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
        self.meters[f"{feature_name}_dense/hardcore_mi"].add(unregularized_loss.item())
        self.meters[f"{feature_name}_dense/soften_mi"].add(regularized_loss.item())

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
        self._infonce_criterion2 = SupConLoss4()  # adding a soften loss

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(EncoderDenseMixupContrastEpocher, self)._configure_meters(meters)
        config = get_config(scope="base")
        # todo: adding try except within a context manager
        for f in config["ProjectorParams"]["GlobalParams"]["feature_names"] or []:
            m = "global"
            meters.register_meter(f"{f}_{m}/hardcode_mi", AverageValueMeter(), )
            meters.register_meter(f"{f}_{m}/mixup_mi", AverageValueMeter(), )
        for f in config["ProjectorParams"]["DenseParams"]["feature_names"] or []:
            m = "dense"
            meters.register_meter(f"{f}_{m}/hardcode_mi", AverageValueMeter(), )
            meters.register_meter(f"{f}_{m}/mixup_mi", AverageValueMeter(), )

        return meters

    def run(self, *args, **kwargs):
        with FeatureExtractor(self._model, self._feature_position) as self._fextractor_mixup:  # noqa
            logger.debug(f"create feature extractor with mixup for {', '.join(self._feature_position)} ")
            return super(EncoderDenseMixupContrastEpocher, self).run(*args, **kwargs)  # noqa

    def forward_pass(self, *args, **kwargs):
        self._fextractor_mixup.clear()
        return super(EncoderDenseMixupContrastEpocher, self).forward_pass(*args, **kwargs)

    def regularization(self, *, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int,
                       label_group: List[str], partition_group: List[str], unlabeled_image=None,
                       unlabeled_image_tf=None, **kwargs):
        def prepare_mixup():
            # creating mixup images and labels =====
            b, c, h, w = unlabeled_image.shape
            rand_index = torch.Tensor(np.random.permutation(list(range(b)))).long().to(self.device)
            label = torch.from_numpy(LabelEncoder().fit(partition_group).transform(partition_group)).to(self.device)
            oh_label = class2one_hot(label, C=4).float()

            mixed_image, mixed_oh_target = generate_mixup_image(image_list1=unlabeled_image_tf,
                                                                image_list2=unlabeled_image_tf[rand_index],
                                                                label_list1=oh_label, label_list2=oh_label[rand_index])
            # pass the mixup image to the network
            with self._fextractor_mixup.enable_register(), self._fextractor.disable_register():
                _ = self._model(mixed_image)

            register_variable(name="oh_target", object_=oh_label)
            register_variable(name="mixed_oh_target", object_=mixed_oh_target)

        prepare_mixup()

        reg_loss = super(EncoderDenseMixupContrastEpocher, self).regularization(
            unlabeled_tf_logits=unlabeled_tf_logits, unlabeled_logits_tf=unlabeled_logits_tf,
            seed=seed, label_group=label_group, partition_group=partition_group, unlabeled_image=unlabeled_image,
            unlabeled_image_tf=unlabeled_image_tf, **kwargs)

        return reg_loss

    def _dense_infonce_for_decoder(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                                   label_group):
        raise RuntimeError(f"{self.__class__.__name__} does not support contrasting on dense feature from decoder")

    def _regularization(self, *, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, label_group,
                        partition_group, **kwargs):
        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2

        losses = [
            self.generate_infonce(
                feature_name=n, features=f, mixup_feature=mixup, projector=p, seed=seed,
                partition_group=partition_group,
                label_group=label_group) for n, f, mixup, p in
            zip(self._fextractor.feature_names, unl_extractor(self._fextractor, n_uls=n_uls), self._fextractor_mixup,
                self._projectors_wrapper)
        ]
        reg_loss = weighted_average_iter(losses, self._feature_importance)
        self.meters["mi"].add(-reg_loss.item())
        self.meters["individual_mis"].add(**dict(zip(
            [f"{p}|{i}" for i, p in enumerate(self._feature_position)],
            [-x.item() for x in losses]
        )))
        return reg_loss

    def generate_infonce(self, *, feature_name, features, mixup_feature=None, projector, seed, partition_group,
                         label_group) -> Tensor:
        # note that mixup_feature is created from `unlabeled_image_tf`
        proj_tf_feature, proj_feature_tf = self.unlabeled_projection(
            unl_features=features, projector=projector, seed=seed
        )
        # manually create the projected mixup features.
        proj_mixup_feature = projector(mixup_feature)

        if isinstance(projector, ProjectionHead):
            # it goes to **global** representation here.
            return self._global_infonce(
                feature_name=feature_name,
                proj_tf_feature=proj_tf_feature,
                proj_feature_tf=proj_feature_tf,
                proj_feature_mixup=proj_mixup_feature,
                partition_group=partition_group,
                label_group=label_group
            )
        # it goes to a **dense** representation on pixels
        return self._dense_based_infonce(
            feature_name=feature_name,
            proj_tf_feature=proj_tf_feature,
            proj_feature_tf=proj_feature_tf,
            proj_feature_mixup=proj_mixup_feature,
            partition_group=partition_group,
            label_group=label_group
        )

    def _global_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, proj_feature_mixup=None,
                        partition_group, label_group) -> Tensor:
        unregularized_loss = super()._global_infonce(
            feature_name=feature_name, proj_tf_feature=proj_tf_feature,
            proj_feature_tf=proj_feature_tf, partition_group=partition_group,
            label_group=label_group
        )

        oh_target = get_variable(name="oh_target")
        mixed_oh_target = get_variable(name="mixed_oh_target")
        one2one_coef = generate_similarity_weightmatrix_based_on_labels(
            label_dist1=oh_target,
            label_dist2=oh_target,
        )
        # two2two_coef = generate_similarity_weightmatrix_based_on_labels(
        #     label_dist1=mixed_oh_target,
        #     label_dist2=mixed_oh_target
        # )
        two2two_coef = None

        cross_coef = generate_similarity_weightmatrix_based_on_labels(
            label_dist1=oh_target,
            label_dist2=mixed_oh_target,
        )
        self._infonce_criterion2: SupConLoss4
        regularized_loss = self._infonce_criterion2(
            proj_feat1=proj_tf_feature, proj_feat2=proj_feature_mixup,
            one2one_weight=one2one_coef, two2two_weight=two2two_coef, one2two_weight=cross_coef,
        )

        config = get_config(scope="base")
        reg_weight = float(config["ProjectorParams"]["GlobalParams"]["softweight"])
        self.meters[f"{feature_name}_global/hardcode_mi"].add(unregularized_loss.item())
        self.meters[f"{feature_name}_global/mixup_mi"].add(regularized_loss.item())

        return unregularized_loss + reg_weight * regularized_loss

    def _dense_based_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, proj_feature_mixup=None,
                             partition_group, label_group) -> Tensor:
        if "Conv" in feature_name:
            # this is the dense feature from encoder
            return self._dense_infonce_for_encoder(
                feature_name=feature_name,
                proj_tf_feature=proj_tf_feature,
                proj_feature_tf=proj_feature_tf,
                proj_feature_mixup=proj_feature_mixup,
                partition_group=partition_group,
                label_group=label_group
            )
        return self._dense_infonce_for_decoder(
            feature_name=feature_name,
            proj_tf_feature=proj_tf_feature,
            proj_feature_tf=proj_feature_tf,
            partition_group=partition_group,
            label_group=label_group
        )

    def _dense_infonce_for_encoder(self, *, feature_name, proj_tf_feature, proj_feature_tf, proj_feature_mixup=None,
                                   partition_group, label_group):
        # get unregularized loss based on index of the dense feature map.
        unregularized_loss = super()._dense_infonce_for_encoder(feature_name=feature_name,
                                                                proj_tf_feature=proj_tf_feature,
                                                                proj_feature_tf=proj_feature_tf,
                                                                partition_group=partition_group,
                                                                label_group=label_group)
        # repeat a bit
        assert "Conv" in feature_name, feature_name
        b, c, *hw = proj_tf_feature.shape
        proj_tf_feature, proj_feature_tf = self._reshape_dense_feature(proj_tf_feature, proj_feature_tf)
        proj_feature_mixup = proj_feature_mixup.view(b, c, -1).permute(0, 2, 1)

        b, hw, c = proj_feature_tf.shape

        if not (is_normalized(proj_feature_tf, dim=2) and is_normalized(proj_feature_tf, dim=2)):
            proj_feature_tf = Normalize(dim=2)(proj_feature_tf)
            proj_tf_feature = Normalize(dim=2)(proj_tf_feature)
            proj_feature_mixup = Normalize(dim=2)(proj_feature_mixup)

        proj_feature_tf, proj_tf_feature, proj_feature_mixup = proj_feature_tf.reshape(-1, c), \
                                                               proj_tf_feature.reshape(-1, c), \
                                                               proj_feature_mixup.reshape(-1, c)

        one2one_coef = generate_similarity_masks(proj_feature_tf, proj_feature_tf)
        two2two_coef = None
        cross_coef = generate_similarity_masks(proj_feature_tf, proj_feature_mixup)
        regularized_loss = self._infonce_criterion2(
            proj_feat1=proj_tf_feature, proj_feat2=proj_feature_mixup,
            one2one_weight=one2one_coef, two2two_weight=two2two_coef, one2two_weight=cross_coef, )
        self.meters[f"{feature_name}_dense/hardcode_mi"].add(unregularized_loss.item())
        self.meters[f"{feature_name}_dense/mixup_mi"].add(regularized_loss.item())

        config = get_config(scope="base")
        reg_weight = float(config["ProjectorParams"]["DenseParams"]["softweight"])

        return unregularized_loss + reg_weight * regularized_loss
