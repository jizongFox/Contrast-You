import math
from functools import lru_cache

import numpy as np
import torch
from contrastyou.losses.contrast_loss import SupConLoss3, is_normalized
from contrastyou.losses.iic_loss import _ntuple  # noqa
from contrastyou.projectors.nn import Normalize
from deepclustering2.configparser._utils import get_config
from deepclustering2.meters2 import MeterInterface, AverageValueMeter
from deepclustering2.type import T_loss
from loguru import logger
from torch import Tensor

from .comparable import InfoNCEEpocher
from .._utils import ContrastiveProjectorWrapper


class ProposedEpocher1(InfoNCEEpocher):

    def _init(self, *, reg_weight: float, projectors_wrapper: ContrastiveProjectorWrapper = None,
              infoNCE_criterion: T_loss = None, kernel_size: int = 1, margin: int = 3, neigh_weight: float = None,
              **kwargs):
        super(ProposedEpocher1, self)._init(reg_weight=reg_weight, projectors_wrapper=projectors_wrapper,
                                            infoNCE_criterion=infoNCE_criterion, **kwargs)
        self._kernel_size = kernel_size  # noqa
        self._margin = margin  # noqa
        assert 0 <= neigh_weight <= 1, neigh_weight
        self._neigh_weight = neigh_weight  # noqa
        if self._neigh_weight == 1:
            logger.debug("{}, only considering neighor term", self.__class__.__name__, )
        elif self._neigh_weight == 0:
            logger.debug("{}, only considering content term", self.__class__.__name__, )
        else:
            logger.debug("{}, considers neighor term: {} and content term: {}", self.__class__.__name__,
                         self._neigh_weight, 1 - self._neigh_weight)

    def _dense_infonce_for_decoder(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                                   label_group):
        b, c, *hw = proj_feature_tf.shape
        nearby_mask = self.generate_relation_masks(
            output_size=(math.sqrt(np.product(hw)), math.sqrt(hw)), kernel_size=self._kernel_size, margin=self._margin
        )
        sim_mask = self.generate_similarity_masks(norm_feature1=proj_feature_tf, norm_feature2=proj_tf_feature)

    @lru_cache()
    def generate_relation_masks(self, output_size, kernel_size=1, margin=1) -> Tensor:
        _pair = _ntuple(2)
        output_size = _pair(output_size)
        size = output_size[0] * output_size[1]
        mask = torch.zeros(size, size, dtype=torch.float, device=self._device)
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                relation = torch.zeros(*output_size, dtype=torch.float, device=self._device)
                relation[
                max(i - kernel_size - margin, 0):i + kernel_size + margin,
                max(j - kernel_size - margin, 0):j + kernel_size + margin
                ] = -1
                relation[max(i - kernel_size, 0):i + kernel_size, max(j - kernel_size, 0):j + kernel_size] = 1
                mask[i * output_size[0] + j] = relation.view(1, -1)
        return mask

    @torch.no_grad()
    def generate_similarity_masks(self, norm_feature1: Tensor, norm_feature2: Tensor):
        sim_matrix = norm_feature1.bmm(norm_feature2.transpose(2, 1))  # ranging from -1 to 1
        sim_matrix -= sim_matrix.min()
        sim_matrix /= 2
        for i in range(sim_matrix.size(1)):
            sim_matrix[:, i, i] = 1

        new_matrix = torch.zeros_like(sim_matrix).fill_(-1)
        new_matrix[sim_matrix > 0.95] = 1
        new_matrix[sim_matrix < 0.65] = 0
        return new_matrix


class ProposedEpocher2(InfoNCEEpocher):

    def _assertion(self):
        from contrastyou.losses.contrast_loss import SupConLoss3
        if not isinstance(self._infonce_criterion, SupConLoss3):
            raise RuntimeError(f"{self.__class__.__name__} only support `SupConLoss3`, "
                               f"given {type(self._infonce_criterion)}")
        super(InfoNCEEpocher, self)._assertion()


class ProposedEpocher3(InfoNCEEpocher):

    def _init(self, *, reg_weight: float, projectors_wrapper: ContrastiveProjectorWrapper = None,
              infoNCE_criterion: T_loss = None, **kwargs):
        super()._init(reg_weight=reg_weight, projectors_wrapper=projectors_wrapper, infoNCE_criterion=infoNCE_criterion,
                      **kwargs)
        self._infonce_criterion2 = SupConLoss3()

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(ProposedEpocher3, self)._configure_meters(meters)
        config = get_config(scope="base")
        for f in config["ProjectorParams"]["GlobalParams"]["feature_names"]:
            m = "global"
            meters.register_meter(f"{f}_{m}/origin_mi", AverageValueMeter(), )
            meters.register_meter(f"{f}_{m}/weight_mi", AverageValueMeter(), )
        for f in config["ProjectorParams"]["DenseParams"]["feature_names"]:
            m = "dense"
            meters.register_meter(f"{f}_{m}/origin_mi", AverageValueMeter(), )
            meters.register_meter(f"{f}_{m}/weight_mi", AverageValueMeter(), )
        return meters

    def _global_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                        label_group) -> Tensor:
        unregularized_loss = super()._global_infonce(feature_name=feature_name, proj_tf_feature=proj_tf_feature,
                                                     proj_feature_tf=proj_feature_tf, partition_group=partition_group,
                                                     label_group=label_group)
        labels = self.global_label_generator(self.__encoder_method_name__)(partition_list=partition_group,
                                                                           patient_list=label_group)
        # labels = torch.Tensor(labels).to(device=self.device)
        # o_mask = torch.eq(labels[..., None], labels[None, ...]).float()
        sim_coef = self.generate_similarity_masks(proj_tf_feature, proj_feature_tf)
        self._infonce_criterion2: SupConLoss3
        regularized_loss = self._infonce_criterion2(proj_tf_feature, proj_feature_tf,
                                                    pos_weight=sim_coef)
        self.meters[f"{feature_name}_global/origin_mi"].add(unregularized_loss.item())
        self.meters[f"{feature_name}_global/weight_mi"].add(regularized_loss.item())

        return unregularized_loss + 0.01 * regularized_loss

    def _dense_infonce_for_encoder(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                                   label_group):
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
        sim_coef = self.generate_similarity_masks(proj_feature_tf, proj_tf_feature)
        regularized_loss = self._infonce_criterion2(proj_tf_feature, proj_feature_tf,
                                                    pos_weight=sim_coef)
        self.meters[f"{feature_name}_dense/origin_mi"].add(unregularized_loss.item())
        self.meters[f"{feature_name}_dense/weight_mi"].add(regularized_loss.item())
        return unregularized_loss + 0.01 * regularized_loss

    @torch.no_grad()
    def generate_similarity_masks(self, norm_feature1: Tensor, norm_feature2: Tensor):
        sim_matrix = norm_feature1.mm(norm_feature2.transpose(1, 0))  # ranging from -1 to 1
        sim_matrix -= -1
        sim_matrix /= 2
        return sim_matrix
