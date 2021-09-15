# this file implements the multicore loss that lets a class having multiple prototype representation
import typing as t

import torch
from loguru import logger
from torch import nn, Tensor

from contrastyou.losses.kl import KL_div
from contrastyou.utils import simplex, one_hot, class_name
from ._base import LossClass


class MultiCoreKL(nn.Module, LossClass[Tensor]):
    def __init__(self, groups: t.List[t.List[int]]):
        super().__init__()
        self._groups = groups
        self.kl = KL_div()
        logger.trace(f"{class_name(self)} created with groups: {groups}")

    def forward(self, predict_simplex: Tensor, onehot_target: Tensor):
        assert simplex(predict_simplex) and one_hot(onehot_target)
        reduced_simplex = self.reduced_simplex(predict_simplex)
        loss = self.kl(reduced_simplex, onehot_target)
        return loss

    @property
    def groups(self) -> t.List[t.List[int]]:
        return self._groups

    def reduced_simplex(self, predict_simplex: Tensor):
        reduced_simplex = torch.cat([predict_simplex[:, i].sum(1, keepdim=True) for i in self._groups], dim=1)
        return reduced_simplex
