# this file implements the multicore loss that lets a class having multiple prototype representation
import typing as t
from abc import abstractmethod

import torch
from loguru import logger
from torch import nn, Tensor

from contrastyou.losses.kl import KL_div, Entropy
from contrastyou.utils import simplex, one_hot, class_name
from ._base import LossClass


class GradientReverse(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output


def scale_grad(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)


class OverSegmentedLoss(nn.Module, LossClass[Tensor]):
    kl: KL_div

    @abstractmethod
    def reduced_simplex(self, predict_simplex: Tensor) -> Tensor:
        pass


class MultiCoreKL(OverSegmentedLoss):
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


class AdaptiveOverSegmentedLoss(OverSegmentedLoss):

    def __init__(self, input_num_classes: int, output_num_classes: int, device: str, entropy_decay=1e-3) -> None:
        super().__init__()
        self.kl = KL_div()
        self.entroy = Entropy()
        self._entropy_decay = entropy_decay
        self._input_num_classes = input_num_classes
        self._output_num_classes = output_num_classes
        self._translate_matrix = nn.Parameter(
            torch.randn(self._input_num_classes, self._output_num_classes, device=device)
        )

        logger.trace(f"{class_name(self)} created with  matrix shape: {self._translate_matrix.shape}. ")

    def forward(self, predict_simplex: Tensor, onehot_target: Tensor):
        assert simplex(predict_simplex) and one_hot(onehot_target)
        reduced_simplex = self.reduced_simplex(predict_simplex)
        loss = self.kl(reduced_simplex, onehot_target) + \
               self.entroy(self._translate_matrix.softmax(1)) * self._entropy_decay
        return loss

    def reduced_simplex(self, predict_simplex: Tensor) -> Tensor:
        b, c, *_ = predict_simplex.shape
        reduced_simplex = predict_simplex.moveaxis(1, -1) @ scale_grad(self._translate_matrix.softmax(1), 1)
        return reduced_simplex.moveaxis(-1, 1)
