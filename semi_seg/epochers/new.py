from functools import lru_cache

import torch
from torch import Tensor

from contrastyou.helper import weighted_average_iter, average_iter
from contrastyou.losses.iic_loss import _ntuple
from contrastyou.projectors.heads import ProjectionHead, LocalProjectionHead
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.meters2 import AverageValueMeter, MeterInterface, MultipleAverageValueMeter
from deepclustering2.type import T_loss
from semi_seg._utils import ContrastiveProjectorWrapper
from .base import TrainEpocher
from .helper import unl_extractor


class NewEpocher(TrainEpocher):
    """This epocher is going to do feature clustering on UNet intermediate layers, instead of using MI"""
    only_with_labeled_data = False

    def init(self, *, reg_weight: float, projectors_wrapper: ContrastiveProjectorWrapper = None,
             infoNCE_criterion: T_loss = None, kernel_size=1, margin=3, **kwargs):
        assert projectors_wrapper is not None and infoNCE_criterion is not None, (projectors_wrapper, infoNCE_criterion)
        super().init(reg_weight=reg_weight, **kwargs)
        self._projectors_wrapper: ContrastiveProjectorWrapper = projectors_wrapper  # noqa
        self._infonce_criterion: T_loss = infoNCE_criterion  # noqa
        self._kernel_size = kernel_size
        self._margin = margin

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("mi", AverageValueMeter())
        meters.register_meter("individual_mis", MultipleAverageValueMeter())
        return meters

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, label_group,
                       partition_group, *args, **kwargs):
        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2

        def generate_infonce(features, projector):
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
                labels = self.global_label_generator(partition_list=partition_group, patient_list=label_group)
                return self._infonce_criterion(norm_feature_tf, norm_tf_feature, target=labels)


            elif isinstance(projector, LocalProjectionHead):
                norm_tf_feature, norm_feature_tf = proj_tf_feature, proj_feature_tf
                b, c, *hw = norm_tf_feature.shape
                norm_tf_feature = norm_tf_feature.view(b, c, -1).permute(0, 2, 1)
                norm_feature_tf = norm_feature_tf.view(b, c, -1).permute(0, 2, 1)

                mask = self.generate_relation_masks(proj_feature_tf.shape[-2:], kernel_size=self._kernel_size,
                                                    margin=self._margin)

                return average_iter(
                    [self._infonce_criterion(x, y, mask=mask) for x, y in zip(norm_tf_feature, norm_feature_tf)]
                )
            else:
                raise NotImplementedError(type(projector))

        losses = [generate_infonce(f, p) for f, p in zip(unl_extractor(self._fextractor, n_uls=n_uls),
                                                         self._projectors_wrapper)]

        reg_loss = weighted_average_iter(losses, self._feature_importance)
        self.meters["mi"].add(-reg_loss.item())
        self.meters["individual_mis"].add(**dict(zip(
            self._feature_position,
            [-x.item() for x in losses]
        )))
        return reg_loss

    @property  # noqa
    @lru_cache()
    def global_label_generator(self):
        from contrastyou.epocher._utils import GlobalLabelGenerator  # noqa
        return GlobalLabelGenerator()

    @property  # noqa
    @lru_cache()
    def local_label_generator(self):
        from contrastyou.epocher._utils import LocalLabelGenerator  # noqa
        return LocalLabelGenerator()

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
