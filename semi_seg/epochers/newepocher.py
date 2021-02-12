from functools import lru_cache
from typing import Tuple

import torch
from loguru import logger
from torch import Tensor

from contrastyou.helper import average_iter
from contrastyou.losses.contrast_loss import is_normalized, SupConLoss2
from contrastyou.losses.iic_loss import _ntuple  # noqa
from contrastyou.projectors.heads import ProjectionHead, DenseProjectionHead
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.type import T_loss
from semi_seg._utils import ContrastiveProjectorWrapper
from .comparable import _InfoNCEBasedEpocher


class NewEpocher(_InfoNCEBasedEpocher):
    """This epocher takes binary masks defined as the two priors
    infonce_criterion should be set as `SupConLoss2`
    """

    def _init(self, *, reg_weight: float, projectors_wrapper: ContrastiveProjectorWrapper = None,
              infoNCE_criterion: T_loss = None, kernel_size: int = 1, margin: int = 3, neigh_weight: float = None,
              **kwargs):
        super(NewEpocher, self)._init(reg_weight=reg_weight, projectors_wrapper=projectors_wrapper,
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
        self._infonce_criterion: SupConLoss2

    def _assertion(self):
        if not isinstance(self._infonce_criterion, SupConLoss2):
            raise RuntimeError(f"{self.__class__.__name__} only support `SupConLoss2`, "
                               f"given {type(self._infonce_criterion)}")
        super(NewEpocher, self)._assertion()

    def generate_infonce(self, *, feature_name, features, projector, seed, partition_group, label_group):

        proj_tf_feature, proj_feature_tf = self.unlabeled_projection(unl_features=features, projector=projector,
                                                                     seed=seed)
        # normalization and label generation goes differently here.
        if isinstance(projector, ProjectionHead):
            norm_tf_feature, norm_feature_tf = proj_tf_feature, proj_feature_tf
            assert len(norm_tf_feature.shape) == 2, norm_tf_feature.shape
            assert is_normalized(norm_feature_tf) and is_normalized(norm_feature_tf)
            labels = self.global_label_generator(self.__encoder_method_name__)(partition_list=partition_group,
                                                                               patient_list=label_group)
            return self._infonce_criterion(norm_feature_tf, norm_tf_feature, target=labels)

        elif isinstance(projector, DenseProjectionHead):
            norm_tf_feature, norm_feature_tf = proj_tf_feature, proj_feature_tf
            b, c, *hw = norm_tf_feature.shape
            norm_tf_feature = norm_tf_feature.view(b, c, -1).permute(0, 2, 1)
            norm_feature_tf = norm_feature_tf.view(b, c, -1).permute(0, 2, 1)

            mask = self.generate_relation_masks(proj_feature_tf.shape[-2:], kernel_size=self._kernel_size,
                                                margin=self._margin)

            position_mi = average_iter(
                [self._infonce_criterion(x, y, mask=mask) for x, y in zip(norm_tf_feature, norm_feature_tf)]
            )
            content_mi = torch.tensor(0.0, dtype=torch.float, device=self.device)
            if self._neigh_weight < 1:
                relative_mask = self.generate_similarity_masks(norm_tf_feature, norm_feature_tf)

                content_mi = average_iter(
                    [self._infonce_criterion(x, y, mask=_mask) for x, y, _mask in
                     zip(norm_tf_feature, norm_feature_tf, relative_mask)]
                )
            return position_mi * self._neigh_weight + content_mi * (1 - self._neigh_weight)
        else:
            raise NotImplementedError(type(projector))

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


class NewEpocher2(NewEpocher):
    """This epocher is going to do feature clustering on UNet intermediate layers, instead of using MI"""

    def generate_infonce(self, *, features, projector, seed, partition_group, label_group):
        unlabeled_features, unlabeled_tf_features = torch.chunk(features, 2, dim=0)
        with FixRandomSeed(seed):
            unlabeled_features_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_features], dim=0)
        assert unlabeled_tf_features.shape == unlabeled_tf_features.shape, \
            (unlabeled_tf_features.shape, unlabeled_tf_features.shape)

        proj_tf_feature, proj_feature_tf = torch.chunk(
            projector(torch.cat([unlabeled_tf_features, unlabeled_features_tf], dim=0)), 2, dim=0
        )
        # normalization and label generation goes differently here.
        if isinstance(projector, ProjectionHead):
            norm_tf_feature, norm_feature_tf = proj_tf_feature, proj_feature_tf
            assert len(norm_tf_feature.shape) == 2, norm_tf_feature.shape
            assert is_normalized(norm_feature_tf) and is_normalized(norm_feature_tf)
            labels = self.global_label_generator(self.__encoder_method_name__)(partition_list=partition_group,
                                                                               patient_list=label_group)
            return self._infonce_criterion(norm_feature_tf, norm_tf_feature, target=labels)

        elif isinstance(projector, DenseProjectionHead):
            norm_tf_feature, norm_feature_tf = proj_tf_feature, proj_feature_tf
            b, c, *hw = norm_tf_feature.shape
            norm_tf_feature = norm_tf_feature.view(b, c, -1).permute(0, 2, 1)
            norm_feature_tf = norm_feature_tf.view(b, c, -1).permute(0, 2, 1)

            pos_mask, neg_mask = self.generate_relation_masks(proj_feature_tf.shape[-2:],
                                                              kernel_size=self._kernel_size,
                                                              margin=self._margin)

            position_mi = average_iter(
                [self._infonce_criterion(x, y, pos_weight=pos_mask, neg_weight=neg_mask) for x, y in
                 zip(norm_tf_feature, norm_feature_tf)]
            )
            content_mi = torch.tensor(0.0, dtype=torch.float, device=self.device)
            if self._neigh_weight < 1:
                relative_mask = self.generate_similarity_masks(norm_tf_feature, norm_feature_tf)

                content_mi = average_iter(
                    [self._infonce_criterion(x, y, mask=_mask) for x, y, _mask in
                     zip(norm_tf_feature, norm_feature_tf, relative_mask)]
                )
            return position_mi * self._neigh_weight + content_mi * (1 - self._neigh_weight)
        else:
            raise NotImplementedError(type(projector))

    @lru_cache()
    def generate_relation_masks(self, output_size, kernel_size=1, margin=1) -> Tuple[Tensor, Tensor]:
        _pair = _ntuple(2)
        output_size = _pair(output_size)
        size = output_size[0] * output_size[1]
        import numpy as np
        from scipy.ndimage import gaussian_filter
        mask = np.zeros([size, size])
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                relation = np.zeros(output_size)
                relation[i, j] = 1
                filtered_relation = gaussian_filter(relation, sigma=kernel_size)
                filtered_relation /= filtered_relation.max()  # noqa
                mask[i * output_size[0] + j] = filtered_relation.reshape(1, -1)
        mask = torch.from_numpy(mask).float().to(self._device)
        return mask, 1 - mask
