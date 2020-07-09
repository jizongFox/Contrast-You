import sys
from typing import Tuple

import torch
from deepclustering.utils import simplex
from termcolor import colored
from torch import Tensor
from torch.nn import functional as F


class IIDSegmentationLoss:
    def __init__(
        self, lamda=1.0, padding=7, eps: float = sys.float_info.epsilon
    ) -> None:
        print(colored(f"Initialize {self.__class__.__name__}.", "green"))
        self.lamda = lamda
        self.padding = padding
        self.eps = eps
        self.torch_vision = torch.__version__

    def __call__(
        self, x_out: Tensor, x_tf_out: Tensor, mask: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        assert x_out.requires_grad and x_tf_out.requires_grad
        if mask is not None:
            assert not mask.requires_grad
        assert simplex(x_out)
        assert x_out.shape == x_tf_out.shape
        bn, k, h, w = x_tf_out.shape
        if mask is not None:
            x_out = x_out * mask
            x_tf_out = x_tf_out * mask

        x_out = x_out.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
        x_tf_out = x_tf_out.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
        # k, k, 2 * half_T_side_dense + 1,2 * half_T_side_dense + 1
        p_i_j = F.conv2d(x_out, weight=x_tf_out, padding=(self.padding, self.padding))
        T_side_dense = self.padding * 2 + 1

        # T x T x k x k
        p_i_j = p_i_j.permute(2, 3, 0, 1)
        p_i_j = p_i_j / p_i_j.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)  # norm

        # symmetrise, transpose the k x k part
        p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0

        # T x T x k x k
        p_i_mat = p_i_j.sum(dim=2, keepdim=True).repeat(1, 1, k, 1)
        p_j_mat = p_i_j.sum(dim=3, keepdim=True).repeat(1, 1, 1, k)

        if self.torch_vision < "1.3.0":
            p_i_j[(p_i_j < self.eps).data] = self.eps
            p_i_mat[(p_i_mat < self.eps).data] = self.eps
            p_j_mat[(p_j_mat < self.eps).data] = self.eps

        # maximise information
        loss = (
                   -p_i_j
                   * (
                       torch.log(p_i_j)
                       - self.lamda * torch.log(p_i_mat)
                       - self.lamda * torch.log(p_j_mat)
                   )
               ).sum() / (T_side_dense * T_side_dense)
        return (
            loss,
            torch.tensor(0, dtype=torch.float, device=x_out.device),
            torch.tensor(0, dtype=torch.float, device=x_out.device),
        )
