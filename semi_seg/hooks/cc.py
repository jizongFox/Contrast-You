# this hook tries to use patch-based Cross Correlation loss on the over-segmentation softmax and the original image.
import torch
from torch import Tensor

from contrastyou.hooks import TrainerHook, EpocherHook
from contrastyou.losses.cc import CCLoss
from contrastyou.meters import MeterInterface, AverageValueMeter


class CrossCorrelationHook(TrainerHook):

    def __init__(self, *, hook_name: str = "cc", weight: float, kernel_size: int, device):
        super().__init__(hook_name=hook_name)
        self._kernel_size = kernel_size
        self._cc_criterion = CCLoss(win=(self._kernel_size, self._kernel_size), device=device)
        self._weight = weight

    def __call__(self, **kwargs):
        return _CrossCorrelationEpocherHook(name=self._hook_name, criterion=self._cc_criterion, weight=self._weight)


class _CrossCorrelationEpocherHook(EpocherHook):

    def __init__(self, *, name: str = "cc", criterion: CCLoss, weight: float, ) -> None:
        super().__init__(name=name)
        self.criterion = criterion
        self.weight = weight

    def configure_meters(self, meters: MeterInterface):
        meters.register_meter("loss", AverageValueMeter())
        return meters

    def _call_implementation(self, unlabeled_image_tf: Tensor, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor,
                             **kwargs):
        # assert unlabeled_image_tf.max() <= 1 and unlabeled_image_tf.min() >= 0
        diff_image = self.diff(unlabeled_image_tf)
        diff_tf_softmax = self.diff(unlabeled_tf_logits)
        diff_softmax_tf = self.diff(unlabeled_logits_tf)
        loss = self.criterion(diff_image, diff_tf_softmax) + self.criterion(diff_image, diff_softmax_tf)
        self.meters["loss"].add(loss.item())
        return loss

    @staticmethod
    def diff(image: Tensor):
        b, c, h, w = image.shape
        dx = image - torch.roll(image, shifts=1, dims=2)
        dy = image - torch.roll(image, shifts=1, dims=3)
        d = torch.sqrt(dx.pow(2) + dy.pow(2))
        return torch.mean(d, dim=[1], keepdims=True)  # noqa