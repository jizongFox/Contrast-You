# this hook tries to use patch-based Cross Correlation loss on the over-segmentation softmax and the original image.

import torch
from loguru import logger
from torch import Tensor

from contrastyou.hooks import TrainerHook, EpocherHook
from contrastyou.losses.cc import CCLoss
from contrastyou.losses.kl import Entropy
from contrastyou.meters import MeterInterface, AverageValueMeter
from contrastyou.utils import class_name


class CrossCorrelationHook(TrainerHook):

    def __init__(self, *, hook_name: str = "cc", weight: float, kernel_size: int, device):
        super().__init__(hook_name=hook_name)
        self._kernel_size = kernel_size
        self._cc_criterion = CCLoss(win=(self._kernel_size, self._kernel_size), device=device)
        self._weight = weight
        logger.debug(f"Creating {class_name(self)} with weight: {self._weight} and kernel_size: {self._kernel_size}.")

    def __call__(self, **kwargs):
        return _CrossCorrelationEpocherHook(name=self._hook_name, criterion=self._cc_criterion, weight=self._weight)


class _CrossCorrelationEpocherHook(EpocherHook):

    def __init__(self, *, name: str = "cc", criterion: CCLoss, weight: float, ) -> None:
        super().__init__(name=name)
        self.criterion = criterion
        self.weight = weight
        self._ent_criterion = Entropy(reduction="none")

    def configure_meters_given_epocher(self, meters: MeterInterface):
        meters.register_meter("loss", AverageValueMeter())
        return meters

    def _call_implementation(self, unlabeled_image_tf: Tensor, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor,
                             **kwargs):
        diff_image = self.diff(unlabeled_image_tf)
        diff_tf_softmax = self._ent_criterion(unlabeled_tf_logits.softmax(1)).unsqueeze(1)
        loss = self.criterion(
            self.norm(diff_tf_softmax),
            self.norm(diff_image)
        )
        self.meters["loss"].add(loss.item())
        return loss * self.weight

    @staticmethod
    def norm(image: Tensor):
        min_, max_ = image.min().detach(), image.max().detach()
        image = image - min_
        image = image / (max_ - min_ + 1e-6)
        return (image - 0.5) * 2

    @staticmethod
    def diff(image: Tensor):
        assert image.dim() == 4
        dx = image - torch.roll(image, shifts=1, dims=2)
        dy = image - torch.roll(image, shifts=1, dims=3)
        d = torch.sqrt(dx.pow(2) + dy.pow(2))
        return torch.mean(d, dim=[1], keepdims=True)  # noqa
