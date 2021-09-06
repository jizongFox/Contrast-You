# this file implements the multicore loss that lets a class having multiple prototype representation
import typing as t

import torch
from torch import nn, Tensor

from contrastyou.losses.kl import KL_div, Entropy
from contrastyou.utils import simplex, one_hot, class2one_hot


class MultiCoreKL(nn.Module):
    def __init__(self, groups: t.List[t.List[int]], entropy_weight=0.01, **kwargs):
        super().__init__()
        self._groups = groups
        self._ent_weight = entropy_weight
        self._kl = KL_div()
        self._ent_criterion = Entropy()

    def forward(self, predict_simplex: Tensor, onehot_target: Tensor):
        assert simplex(predict_simplex) and one_hot(onehot_target)
        reduced_simplex = self.reduced_simplex(predict_simplex)
        loss = self._kl(reduced_simplex, onehot_target)
        ent_loss = self._ent_criterion(predict_simplex)
        return loss + self._ent_weight * ent_loss

    @property
    def groups(self) -> t.List[t.List[int]]:
        return self._groups

    def reduced_simplex(self, predict_simplex: Tensor):
        reduced_simplex = torch.cat([predict_simplex[:, i].sum(1, keepdim=True) for i in self._groups], dim=1)
        return reduced_simplex


if __name__ == '__main__':
    output = torch.randn(100, 15, ).softmax(1)
    target = torch.randint(0, 3, (100,))
    groups = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14]]
    criterion = MultiCoreKL(groups=groups, entropy_weight=0.1)
    loss = criterion(output, class2one_hot(target, 3))
