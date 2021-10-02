import torch
from loguru import logger
from torch import Tensor

from contrastyou.arch import UNet
from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.losses.discreteMI import IIDSegmentationLoss
from contrastyou.losses.kl import Entropy
from contrastyou.meters import AverageValueMeter
from contrastyou.utils import class_name
from semi_seg.hooks.utils import meter_focus

decoder_names = UNet.decoder_names
encoder_names = UNet.encoder_names

entropy_criterion = Entropy(reduction="none")


class IIDSegmentationTrainerHook(TrainerHook):

    def __init__(self, *, hook_name: str = "midl_hook", weight: float = 1.0) -> None:
        super().__init__(hook_name=hook_name)
        self._weight = weight
        logger.debug(f"Created {class_name(self)} with name: {self._hook_name}, weight: {self._weight}.")

    def __call__(self):
        return _IIDSegmentationEpochHook(name=self._hook_name, weight=self._weight)

        # return IIDSegmentationLoss


class _IIDSegmentationEpochHook(EpocherHook):

    def __init__(self, *, name: str, weight: float) -> None:
        super().__init__(name=name)
        self._weight = weight
        self._criterion = IIDSegmentationLoss(padding=0)

    @meter_focus
    def configure_meters(self, meters):
        meters.register_meter("mi", AverageValueMeter())

    @meter_focus
    def __call__(self, *, unlabeled_tf_logits, unlabeled_logits_tf, **kwargs):
        if self._weight == 0:
            self.meters["mi"].add(0)
            return torch.tensor(0, device=unlabeled_logits_tf.device, dtype=unlabeled_logits_tf.dtype)
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


def IMSAT_loss(prediction: Tensor):
    pred = prediction.moveaxis(0, 1).reshape(prediction.shape[1], -1)
    margin = pred.mean(1, keepdims=True)

    mi = -entropy_criterion(pred.t()).mean() + entropy_criterion(margin.t()).mean()

    return -mi


class _IMSATEpochHook(EpocherHook):

    def __init__(self, *, name: str, weight: float) -> None:
        super().__init__(name=name)
        self._weight = weight

    @meter_focus
    def configure_meters(self, meters):
        meters.register_meter("mi", AverageValueMeter())

    @meter_focus
    def __call__(self, *, unlabeled_logits_tf, **kwargs):
        unlabeled_tf_softmax = unlabeled_logits_tf.softmax(1)

        loss = IMSAT_loss(unlabeled_tf_softmax)
        self.meters["mi"].add(loss.item())
        return loss * self._weight
