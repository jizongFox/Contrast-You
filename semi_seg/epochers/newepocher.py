from functools import lru_cache

import torch
from loguru import logger
from torch import Tensor

from contrastyou.epocher._utils import preprocess_input_with_single_transformation  # noqa
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation  # noqa
from contrastyou.epocher._utils import write_predict, write_img_target  # noqa
from contrastyou.helper import weighted_average_iter, average_iter
from contrastyou.losses.contrast_loss import is_normalized
from contrastyou.losses.iic_loss import _ntuple  # noqa
from contrastyou.projectors.heads import ProjectionHead, LocalProjectionHead
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats as disable_bn  # noqa
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.meters2 import AverageValueMeter, MeterInterface, MultipleAverageValueMeter
from deepclustering2.type import T_loss
from semi_seg._utils import ContrastiveProjectorWrapper
from .base import TrainEpocher, EvalEpocher
from .helper import unl_extractor


class EvalEpocherWOEval(EvalEpocher):
    """
    This epocher is set to using the current estimation of batch and without accumulating the statistic
    network in train mode while BN is in disable accumulation mode.
    Usually improves performance with some domain gap
    """

    def _run(self, *args, **kwargs):
        with disable_bn(self._model):  # disable bn accumulation
            return super(EvalEpocherWOEval, self)._run(*args, **kwargs)

    def _set_model_state(self, model) -> None:
        model.train()


class NewEpocher(TrainEpocher):
    """This epocher is going to do feature clustering on UNet intermediate layers, instead of using MI"""

    def init(self, *, reg_weight: float, projectors_wrapper: ContrastiveProjectorWrapper = None,
             infoNCE_criterion: T_loss = None, kernel_size=1, margin=3, neigh_weight: float = None, **kwargs):
        assert projectors_wrapper is not None and infoNCE_criterion is not None, (projectors_wrapper, infoNCE_criterion)
        super().init(reg_weight=reg_weight, **kwargs)
        self._projectors_wrapper: ContrastiveProjectorWrapper = projectors_wrapper  # noqa
        self._infonce_criterion: T_loss = infoNCE_criterion  # noqa
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
                assert is_normalized(norm_feature_tf) and is_normalized(norm_feature_tf)
                labels = self.global_label_generator(partition_list=partition_group, patient_list=label_group)
                return self._infonce_criterion(norm_feature_tf, norm_tf_feature, target=labels)

            elif isinstance(projector, LocalProjectionHead):
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

    @torch.no_grad()
    def generate_similarity_masks(self, norm_feature1: Tensor, norm_feature2: Tensor):
        batch, b, c = norm_feature1.shape
        sim_matrix = norm_feature1.bmm(norm_feature2.transpose(2, 1))  # ranging from -1 to 1
        sim_matrix -= sim_matrix.min()
        sim_matrix /= 2
        for i in range(sim_matrix.size(1)):
            sim_matrix[:, i, i] = 1

        new_matrix = torch.zeros_like(sim_matrix).fill_(-1)
        new_matrix[sim_matrix > 0.95] = 1
        new_matrix[sim_matrix < 0.65] = 0
        return new_matrix
