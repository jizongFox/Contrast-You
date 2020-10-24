import torch
from torch import Tensor

from contrastyou.epocher._utils import preprocess_input_with_single_transformation  # noqa
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation  # noqa
from contrastyou.epocher._utils import write_predict, write_img_target  # noqa
from contrastyou.helper import average_iter, weighted_average_iter
from contrastyou.projectors.heads import ClusterHead  # noqa
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.meters2 import AverageValueMeter, MultipleAverageValueMeter, \
    MeterInterface
from deepclustering2.type import T_loss
from semi_seg._utils import ClusterProjectorWrapper, IICLossWrapper
from .base import TrainEpocher
from .helper import unl_extractor


# noinspection Mypy
class UDATrainEpocher(TrainEpocher):

    def init(self, *, reg_weight: float, reg_criterion: T_loss, ):  # noqa
        super().init(reg_weight=reg_weight)
        self._reg_criterion = reg_criterion  # noqa

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("uda", AverageValueMeter())
        return meters

    # noinspection Mypy
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

    def init(self, *, reg_weight: float, projectors_wrapper: ClusterProjectorWrapper,  # noqa
             IIDSegCriterionWrapper: IICLossWrapper, enforce_matching=False,  # noqa
             **kwargs):  # noqa
        super().init(reg_weight=reg_weight, **kwargs)
        self._projectors_wrapper = projectors_wrapper  # noqa
        self._IIDSegCriterionWrapper = IIDSegCriterionWrapper  # noqa
        self._enforce_matching = enforce_matching  # noqa

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("mi", AverageValueMeter())
        meters.register_meter("individual_mis", MultipleAverageValueMeter())
        return meters

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, *args, **kwargs):
        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2

        def generate_iic(unlabeled_features, projector, criterion):
            unlabeled_features, unlabeled_tf_features = torch.chunk(unlabeled_features, 2, dim=0)

            with FixRandomSeed(seed):
                unlabeled_features_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_features], dim=0)
            assert unlabeled_tf_features.shape == unlabeled_tf_features.shape, \
                (unlabeled_tf_features.shape, unlabeled_tf_features.shape)
            prob1, prob2 = list(
                zip(*[torch.chunk(x, 2, 0) for x in projector(
                    torch.cat([unlabeled_features_tf, unlabeled_tf_features], dim=0)
                )])
            )
            loss = average_iter([criterion(x, y) for x, y in zip(prob1, prob2)])
            return loss

        iic_losses = [generate_iic(f, p, c) for f, p, c in zip(
            unl_extractor(self._fextractor, n_uls=n_uls),
            self._projectors_wrapper,
            self._IIDSegCriterionWrapper
        )]

        reg_loss = weighted_average_iter(iic_losses, self._feature_importance)

        with torch.no_grad():
            self.meters["mi"].add(-reg_loss.item())
            self.meters["individual_mis"].add(**dict(zip(
                self._feature_position,
                [-x.item() for x in iic_losses]
            )))

        return reg_loss


class UDAIICEpocher(IICTrainEpocher):

    def init(self, *, iic_weight: float, uda_weight: float, projectors_wrapper: ClusterProjectorWrapper,  # noqa
             IIDSegCriterionWrapper: IICLossWrapper, reg_criterion: T_loss, enforce_matching=False, **kwargs):  # noqa
        super().init(reg_weight=1.0, projectors_wrapper=projectors_wrapper,
                     IIDSegCriterionWrapper=IIDSegCriterionWrapper, enforce_matching=enforce_matching, **kwargs)
        self._iic_weight = iic_weight  # noqa
        self._cons_weight = uda_weight  # noqa
        self._reg_criterion = reg_criterion  # noqa
        assert self._reg_weight == 1.0, self._reg_weight

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
