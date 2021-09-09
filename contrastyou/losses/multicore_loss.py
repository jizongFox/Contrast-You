# this file implements the multicore loss that lets a class having multiple prototype representation
import typing as t

import torch
from loguru import logger
from torch import nn, Tensor

from contrastyou.losses.kl import KL_div
from contrastyou.utils import simplex, one_hot, class_name


class MultiCoreKL(nn.Module):
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

# class OrthogonalMultiCoreKL(MultiCoreKL):
#
#     def __init__(self, groups: t.List[t.List[int]], entropy_weight=0.0, orthogonal_weight=0.001, **kwargs):
#         super().__init__(groups, entropy_weight, **kwargs)
#         self._orthogonal_weight = orthogonal_weight
#         logger.trace(
#             f"Creating {class_name(self)} with ent_weight: {entropy_weight}, orthogonal_weight: {orthogonal_weight}.")
#
#     def set_fc_layer(self, fc_layer: nn.Module):
#         self._fc = fc_layer
#
#     def forward(self, predict_simplex: Tensor, onehot_target: Tensor):
#         loss = super().forward(predict_simplex, onehot_target)
#         matrix = self.pairwise_matrix(self._fc.weight.squeeze(), self._fc.weight.squeeze())
#         self.orth_loss = matrix.mean() + 1
#         return loss + self._orthogonal_weight * self.orth_loss
#
#     def get_orth_loss(self):
#         try:
#             return self.orth_loss
#         except AttributeError:
#             raise RuntimeError(f"get_orth_loss can only be called after forward.")
#
#     @staticmethod
#     def pairwise_matrix(vec1: Tensor, vec2: Tensor):
#         return vec1 @ vec2.t()
#
#
# if __name__ == '__main__':
#     output = torch.randn(100, 15, ).softmax(1)
#     target = torch.randint(0, 3, (100,))
#     groups = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14]]
#     criterion = MultiCoreKL(groups=groups, entropy_weight=0.1)
#     loss = criterion(output, class2one_hot(target, 3))
