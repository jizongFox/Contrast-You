# this loss is adapted from https://arxiv.org/pdf/2103.03230.pdf
from functools import lru_cache

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from contrastyou.losses import LossClass
from contrastyou.losses.kl import KL_div


class RedundencyCriterion(nn.Module, LossClass[Tensor]):

    def __init__(self, eps: float = 1e-8, symmetric: bool = True) -> None:
        super().__init__()
        self.kl_criterion = KL_div()
        self._eps = eps
        self.symmetric = symmetric

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        k = x_out.shape[1]
        x_out = x_out.swapaxes(0, 1).contiguous()
        x_tf_out = x_tf_out.swapaxes(0, 1).contiguous()
        p_i_j = F.conv2d(
            input=x_out,
            weight=x_tf_out, padding=(0, 0)
        )
        p_i_j = p_i_j - p_i_j.min().detach() + self._eps

        # T x T x k x k
        p_i_j = p_i_j.permute(2, 3, 0, 1)
        p_i_j /= p_i_j.sum(dim=[2, 3], keepdim=True)  # norm

        # symmetrise, transpose the k x k part
        if self.symmetric:
            p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0
        p_i_j /= p_i_j.sum()  # norm
        p_i_j = p_i_j.view(k, k)
        return (-self.onehot_label(k, device=p_i_j.device, dtype=p_i_j.dtype) * p_i_j.log()).mean()

    @lru_cache()
    def onehot_label(self, k, device, dtype):
        label = torch.eye(k, device=device, dtype=dtype)
        return label


if __name__ == '__main__':
    x_output = torch.rand(1, 5, 224, 224, requires_grad=True, device="cuda")
    x_output_tf = torch.rand(1, 5, 224, 224, requires_grad=True, device="cuda")
    optimizer = torch.optim.Adam((x_output, x_output), lr=1e-1)
    for i in range(1000000):
        loss = RedundencyLoss()(x_output.softmax(1), x_output_tf.softmax(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
