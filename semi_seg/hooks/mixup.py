from contextlib import nullcontext
from functools import lru_cache
from typing import Union

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.losses.kl import KL_div
from contrastyou.meters import MeterInterface, AverageValueMeter
from contrastyou.utils import fix_all_seed_within_context, disable_tracking_bn_stats, class2one_hot


def mixup_data(x, y, *, alpha=1.0, device: Union[str, torch.device]):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y


class MixUpTrainHook(TrainerHook):
    def __init__(self, *, hook_name: str, weight: float, enable_bn=True):
        super().__init__(hook_name=hook_name)
        self._weight = weight
        self._enable_bn = enable_bn
        logger.debug(f"created {self.__class__.__name__} with weight: {self._weight} and enable_bn: {self._enable_bn}")

    def __call__(self, **kwargs):
        return _MixUpEpocherHook(name="mix_reg", weight=self._weight, criterion=KL_div(),
                                 enable_bn=self._enable_bn)


class _MixUpEpocherHook(EpocherHook):
    def __init__(self, *, name: str, weight: float, alpha: float = 1.0, criterion, enable_bn: bool) -> None:
        super().__init__(name=name)
        self._weight = weight
        self._alpha = alpha
        self._criterion = criterion
        self._enable_bn = enable_bn

    def configure_meters_given_epocher(self, meters: MeterInterface):
        meters = super(_MixUpEpocherHook, self).configure_meters_given_epocher(meters)
        meters.register_meter("loss", AverageValueMeter())
        return meters

    def _call_implementation(
            self, *,
            labeled_image: Tensor,
            labeled_image_tf: Tensor,
            labeled_target: Tensor,
            labeled_target_tf: Tensor,
            seed: int,
            **kwargs
    ):
        labeled_target_oh = class2one_hot(labeled_target.to(torch.int64), C=self.num_classes).float()
        labeled_target_tf_oh = class2one_hot(labeled_target_tf.to(torch.int64), C=self.num_classes).float()

        with fix_all_seed_within_context(seed):
            mixed_image, mixed_target, = mixup_data(x=torch.cat([labeled_image, labeled_image_tf], dim=0),
                                                    y=torch.cat([labeled_target_oh, labeled_target_tf_oh], dim=0),
                                                    alpha=1, device=self.device)
        with self.bn_context_manger(self._model):
            mixed_logits = self._model(mixed_image)
        reg_loss = self._criterion(mixed_logits.softmax(1), mixed_target.squeeze())
        if self.meters:
            self.meters["loss"].add(reg_loss.item())
        return reg_loss * self._weight

    @property
    def _model(self):
        return self.epocher._model  # noqa

    @property
    def device(self):
        return self.epocher.device

    @property
    def num_classes(self):
        return self.epocher.num_classes

    @property
    @lru_cache()
    def bn_context_manger(self):
        return disable_tracking_bn_stats if not self._enable_bn else nullcontext
