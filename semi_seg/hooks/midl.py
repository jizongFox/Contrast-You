from loguru import logger
from torch import Tensor

from contrastyou.arch import UNet
from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.losses.kl import Entropy
from contrastyou.meters import AverageValueMeter
from contrastyou.utils import class_name

decoder_names = UNet.decoder_names
encoder_names = UNet.encoder_names

entropy_criterion = Entropy(reduction="none", eps=1e-8)


class IIDSegmentationTrainerHook(TrainerHook):

    def __init__(self, *, hook_name: str = "midl_hook", weight: float = 1.0, mi_lambda=1.0) -> None:
        super().__init__(hook_name=hook_name)
        self._weight = weight
        self._mi_lambda = mi_lambda
        logger.debug(f"Created {class_name(self)} with name: {self._hook_name}, weight: {self._weight}, "
                     f"mi_lambda: {mi_lambda}.")

    def __call__(self):
        return _IIDSegmentationEpochHook(name=self._hook_name, weight=self._weight, mi_lambda=self._mi_lambda)


class _IIDSegmentationEpochHook(EpocherHook):

    def __init__(self, *, name: str, weight: float, mi_lambda=1.0) -> None:
        super().__init__(name=name)
        from contrastyou.losses.discreteMI import IIDSegmentationLoss

        self._weight = weight
        self._criterion = IIDSegmentationLoss(padding=0, lamda=mi_lambda)

    def configure_meters_given_epocher(self, meters):
        meters.register_meter("mi", AverageValueMeter())

    def _call_implementation(self, *, unlabeled_tf_logits, unlabeled_logits_tf, **kwargs):
        unlabeled_tf_softmax, unlabeled_softmax_tf = unlabeled_tf_logits.softmax(1), unlabeled_logits_tf.softmax(1)

        loss = self._criterion(unlabeled_tf_softmax, unlabeled_softmax_tf)
        self.meters["mi"].add(loss.item())
        return loss * self._weight


class IMSATTrainHook(TrainerHook):
    def __init__(self, *, hook_name: str = "imsat", weight: float = 0.1):
        super().__init__(hook_name=hook_name)
        self._weight = weight
        logger.debug(f"Created {class_name(self)} with name: {self._hook_name}, weight: {self._weight}.")

    def __call__(self):
        return _IMSATEpochHook(name=self._hook_name, weight=self._weight)


class _IMSATEpochHook(EpocherHook):

    def __init__(self, *, name: str, weight: float) -> None:
        super().__init__(name=name)
        self._weight = weight

    def configure_meters_given_epocher(self, meters):
        meters.register_meter("mi", AverageValueMeter())

    def _call_implementation(self, *, unlabeled_logits_tf, unlabeled_tf_logits: Tensor, **kwargs):
        unlabeled_tf_softmax = unlabeled_tf_logits.softmax(1)
        unlabeled_softmax_tf = unlabeled_logits_tf.softmax(1)

        from contrastyou.losses.discreteMI import imsat_loss
        loss = 0.5 * (imsat_loss(unlabeled_tf_softmax) + imsat_loss(unlabeled_softmax_tf))
        self.meters["mi"].add(loss.item())
        return loss * self._weight
