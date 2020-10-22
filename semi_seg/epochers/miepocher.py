import torch
from torch import Tensor

from contrastyou.epocher._utils import preprocess_input_with_single_transformation  # noqa
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation  # noqa
from contrastyou.epocher._utils import write_predict, write_img_target  # noqa
from contrastyou.helper import average_iter, weighted_average_iter
from contrastyou.trainer._utils import ClusterHead  # noqa
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.meters2 import AverageValueMeter, MultipleAverageValueMeter, \
    MeterInterface
from deepclustering2.type import T_loss
from semi_seg._utils import ProjectorWrapper, IICLossWrapper
from .base import TrainEpocher


class UDATrainEpocher(TrainEpocher):

    def init(self, *, reg_weight: float, reg_criterion: T_loss, **kwargs):  # noqa
        super().init(reg_weight=reg_weight, **kwargs)
        self._reg_criterion = reg_criterion

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("uda", AverageValueMeter())
        return meters

    def regularization(
        self,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed, *args, **kwargs
    ):
        reg_loss = self._reg_criterion(
            unlabeled_tf_logits.softmax(1),
            unlabeled_logits_tf.softmax(1).detach()
        )
        self.meters["uda"].add(reg_loss.item())
        return reg_loss


class IICTrainEpocher(TrainEpocher):

    def init(self, *, reg_weight: float, projectors_wrapper: ProjectorWrapper,  # noqa
             IIDSegCriterionWrapper: IICLossWrapper, enforce_matching=False,  # noqa
             **kwargs):  # noqa
        super().init(reg_weight=reg_weight, **kwargs)
        self._projectors_wrapper = projectors_wrapper
        self._IIDSegCriterionWrapper = IIDSegCriterionWrapper
        self._enforce_matching = enforce_matching

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("mi", AverageValueMeter())
        meters.register_meter("individual_mis", MultipleAverageValueMeter())
        return meters

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, *args, **kwargs):
        feature_names = self._fextractor._feature_names  # noqa
        unlabeled_length = len(unlabeled_tf_logits) * 2
        iic_losses_for_features = []

        for i, (inter_feature, projector, criterion) \
            in enumerate(zip(self._fextractor, self._projectors_wrapper, self._IIDSegCriterionWrapper)):

            unlabeled_features = inter_feature[len(inter_feature) - unlabeled_length:]
            unlabeled_features, unlabeled_tf_features = torch.chunk(unlabeled_features, 2, dim=0)

            if isinstance(projector, ClusterHead):  # features from encoder
                unlabeled_features_tf = unlabeled_features
            else:
                with FixRandomSeed(seed):
                    unlabeled_features_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_features],
                                                        dim=0)
            assert unlabeled_tf_features.shape == unlabeled_tf_features.shape, \
                (unlabeled_tf_features.shape, unlabeled_tf_features.shape)
            prob1, prob2 = list(
                zip(*[torch.chunk(x, 2, 0) for x in projector(
                    torch.cat([unlabeled_features_tf, unlabeled_tf_features], dim=0)
                )])
            )
            _iic_loss_list = [criterion(x, y) for x, y in zip(prob1, prob2)]
            _iic_loss = average_iter(_iic_loss_list)
            iic_losses_for_features.append(_iic_loss)
        reg_loss = weighted_average_iter(iic_losses_for_features, self._feature_importance)
        self.meters["mi"].add(-reg_loss.item())
        self.meters["individual_mis"].add(**dict(zip(
            self._feature_position,
            [-x.item() for x in iic_losses_for_features]
        )))

        return reg_loss


class UDAIICEpocher(IICTrainEpocher):

    def init(self, *, iic_weight: float, uda_weight: float, projectors_wrapper: ProjectorWrapper,  # noqa
             IIDSegCriterionWrapper: IICLossWrapper, reg_criterion: T_loss, enforce_matching=False, **kwargs):  # noqa
        super().init(reg_weight=1.0, projectors_wrapper=projectors_wrapper,
                     IIDSegCriterionWrapper=IIDSegCriterionWrapper, enforce_matching=enforce_matching,
                     **kwargs)
        self._iic_weight = iic_weight
        self._cons_weight = uda_weight
        self._reg_criterion = reg_criterion

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("uda", AverageValueMeter())
        meters.register_meter("iic_weight", AverageValueMeter())
        meters.register_meter("uda_weight", AverageValueMeter())
        return meters

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, *args, **kwargs):
        self.meters["iic_weight"].add(self._iic_weight)
        self.meters["uda_weight"].add(self._cons_weight)
        iic_loss = IICTrainEpocher.regularization(
            self,
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf,
            seed=seed
        )
        cons_loss = UDATrainEpocher.regularization(
            self,  # noqa
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf,
            seed=seed,
        )
        return self._cons_weight * cons_loss + self._iic_weight * iic_loss
