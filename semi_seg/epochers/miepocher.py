from typing import Callable, Iterable

import torch
from torch import Tensor

from contrastyou.epocher._utils import preprocess_input_with_single_transformation  # noqa
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation  # noqa
from contrastyou.epocher._utils import write_predict, write_img_target  # noqa
from contrastyou.helper import weighted_average_iter
from contrastyou.projectors.heads import ClusterHead  # noqa
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.meters2 import AverageValueMeter, MultipleAverageValueMeter, \
    MeterInterface
from deepclustering2.type import T_loss
from .base import TrainEpocher
from .helper import unl_extractor


# noinspection Mypy
class ConsistencyTrainEpocher(TrainEpocher):
    only_with_labeled_data = False

    def init(self, *, reg_weight: float, reg_criterion: T_loss, **kwargs):  # noqa
        super().init(reg_weight=reg_weight, **kwargs)
        self._reg_criterion = reg_criterion  # noqa

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("consistency", AverageValueMeter())
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


class MITrainEpocher(TrainEpocher):
    only_with_labeled_data = False

    def init(self, *, reg_weight: float, mi_estimator_array: Iterable[Callable[[Tensor, Tensor], Tensor]],
             enforce_matching=False, **kwargs):  # noqa
        super().init(reg_weight=reg_weight, **kwargs)
        self._mi_estimator_array = mi_estimator_array  # noqa
        self._enforce_matching = enforce_matching  # noqa

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("mi", AverageValueMeter())
        meters.register_meter("individual_mis", MultipleAverageValueMeter())
        return meters

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, *args, **kwargs):
        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2

        def calculate_iic(unlabeled_features, mi_estimator: Callable[[Tensor, Tensor], Tensor]):
            unlabeled_features, unlabeled_tf_features = torch.chunk(unlabeled_features, 2, dim=0)

            with FixRandomSeed(seed):
                unlabeled_features_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_features], dim=0)
            assert unlabeled_tf_features.shape == unlabeled_tf_features.shape, \
                (unlabeled_tf_features.shape, unlabeled_tf_features.shape)

            loss = mi_estimator(unlabeled_tf_features, unlabeled_features_tf)
            return loss

        iic_losses = [calculate_iic(f, mi) for f, mi in zip(
            unl_extractor(self._fextractor, n_uls=n_uls), self._mi_estimator_array
        )]

        reg_loss = weighted_average_iter(iic_losses, self._feature_importance)

        with torch.no_grad():
            self.meters["mi"].add(-reg_loss.item())
            self.meters["individual_mis"].add(**dict(zip(
                self._feature_position,
                [-x.item() for x in iic_losses]
            )))

        return reg_loss


class ConsistencyMIEpocher(MITrainEpocher):

    def init(self, *, mi_weight: float, consistency_weight: float,  # noqa
             mi_estimator_array: Iterable[Callable[[Tensor, Tensor], Tensor]], reg_criterion: T_loss,  # noqa
             enforce_matching=False, **kwargs):
        super().init(reg_weight=1.0, mi_estimator_array=mi_estimator_array, enforce_matching=enforce_matching, **kwargs)
        self._mi_weight = mi_weight  # noqa
        self._cons_weight = consistency_weight  # noqa
        self._reg_criterion = reg_criterion  # noqa
        assert self._reg_weight == 1.0, self._reg_weight

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("consistency", AverageValueMeter())
        meters.register_meter("mi_weight", AverageValueMeter())
        meters.register_meter("cons_weight", AverageValueMeter())
        return meters

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, *args, **kwargs):
        self.meters["mi_weight"].add(self._mi_weight)
        self.meters["cons_weight"].add(self._cons_weight)
        iic_loss = MITrainEpocher.regularization(
            self,
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf,
            seed=seed
        )
        cons_loss = ConsistencyTrainEpocher.regularization(
            self,  # noqa
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf,
            seed=seed,
        )
        return self._cons_weight * cons_loss + self._mi_weight * iic_loss
