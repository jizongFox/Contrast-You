from contrastyou.arch import UNet
from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.losses.discreteMI import IIDSegmentationLoss
from contrastyou.meters import AverageValueMeter
from semi_seg.hooks.utils import meter_focus

decoder_names = UNet.decoder_names
encoder_names = UNet.encoder_names


class IIDSegmentationTrainerHook(TrainerHook):

    def __init__(self, *, hook_name: str = "midl_hook", weight: float = 1.0) -> None:
        super().__init__(hook_name=hook_name)
        self._weight = weight

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
        unlabeled_tf_softmax, unlabeled_softmax_tf = unlabeled_tf_logits.softmax(1), unlabeled_logits_tf.softmax(1)

        loss = self._criterion(unlabeled_tf_softmax, unlabeled_softmax_tf)
        self.meters["mi"].add(loss.item())
        return loss * self._weight
