import math
import sys
import typing as t

import numpy as np
import torch
from loguru import logger
from termcolor import colored
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from contrastyou.losses._base import LossClass
from contrastyou.utils import average_iter
from contrastyou.utils.general import simplex
from contrastyou.utils.utils import _pair  # noqa
from semi_seg.hooks.midl import entropy_criterion


class IMSATLoss(nn.Module, LossClass[Tensor]):

    def __init__(self, lamda: float = 1.0, eps: float = sys.float_info.epsilon):
        """
        :param eps:
        """
        super().__init__()
        logger.trace(colored(f"Initialize {self.__class__.__name__}.", "green"))
        self.eps = float(eps)
        self.lamda = float(lamda)

    def forward(self, x_out: Tensor, x_tf_out: Tensor = None):
        """
        return the inverse of the MI. if the x_out == y_out, return the inverse of Entropy
        :param x_out:
        :param x_tf_out:
        :return:
        """
        idenity_input = False
        if x_tf_out is None:
            idenity_input = True
            x_tf_out = x_out
        assert len(x_out.shape) == 2, x_out.shape
        assert simplex(x_out), f"x_out not normalized."
        assert simplex(x_tf_out), f"x_tf_out not normalized."
        self.x_out = x_out
        self.x_tf_out = x_tf_out
        if not idenity_input:
            return 0.5 * (imsat_loss(x_out, lamda=self.lamda) + imsat_loss(x_tf_out, lamda=self.lamda))
        return imsat_loss(x_out, lamda=self.lamda)

    def get_joint_matrix(self):
        return compute_joint_2D_with_padding_zeros(self.x_out, self.x_tf_out,
                                                   symmetric=False).squeeze().detach().cpu().numpy()


class IMSATDynamicWeight(IMSATLoss):

    def __init__(self, lamda: float = 1.0, use_dynamic: bool = True,
                 eps: float = sys.float_info.epsilon):
        super().__init__(lamda, eps)
        self.register_buffer("dynamic_weight", torch.tensor(lamda))
        self.use_dynamic_weight = use_dynamic

    def forward(self, x_out: Tensor, **kwargs):
        """
        return the inverse of the MI. if the x_out == y_out, return the inverse of Entropy
        :param x_out:
        :return:
        """
        device, dtype = x_out.device, x_out.dtype
        self.dynamic_weight = self.dynamic_weight.to(device).to(dtype)

        K = x_out.shape[1]
        x_tf_out = x_out
        assert len(x_out.shape) == 2, x_out.shape
        assert simplex(x_out), f"x_out not normalized."
        assert simplex(x_tf_out), f"x_tf_out not normalized."
        self.x_out = x_out
        self.x_tf_out = x_tf_out
        marg, cond = imsat_with_entropy(x_out)
        mi = self.dynamic_weight * marg * -1.0 + cond

        if self.use_dynamic_weight:
            with torch.no_grad():
                increment = (math.log(K) - marg.detach()) * 0.01
                self.dynamic_weight = self.dynamic_weight + increment
        return mi


class IIDLoss(nn.Module, LossClass[t.Tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, lamb: float = 1.0, eps: float = sys.float_info.epsilon):
        """
        :param lamb:
        :param eps:
        """
        super().__init__()
        logger.trace(colored(f"Initialize {self.__class__.__name__}.", "green"))
        self.lamb = float(lamb)
        self.eps = float(eps)

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        """
        return the inverse of the MI. if the x_out == y_out, return the inverse of Entropy
        :param x_out:
        :param x_tf_out:
        :return:
        """
        assert len(x_out.shape) == 2, x_out.shape
        assert simplex(x_out), f"x_out not normalized."
        assert simplex(x_tf_out), f"x_tf_out not normalized."
        _, k = x_out.size()
        p_i_j = compute_joint(x_out, x_tf_out)
        assert p_i_j.size() == (k, k)

        p_i = (p_i_j.sum(dim=1).view(k, 1).expand(k, k))  # p_i should be the mean of the x_out
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

        loss = -p_i_j * (
                torch.log(p_i_j + 1e-10) - self.lamb * torch.log(p_j + 1e-10) - self.lamb * torch.log(p_i + 1e-10)
        )
        loss = loss.sum()
        loss_no_lamb = -p_i_j * (torch.log(p_i_j + 1e-10) - torch.log(p_j + 1e-10) - torch.log(p_i + 1e-10))
        loss_no_lamb = loss_no_lamb.sum()
        return loss, loss_no_lamb, p_i_j


class IIDSegmentationLoss(nn.Module, LossClass[Tensor]):

    def __init__(
            self, lamda=1.0, padding=0, eps: float = 1e-5, symmetric: bool = False,
    ) -> None:
        super(IIDSegmentationLoss, self).__init__()
        logger.trace(f"Initialize {self.__class__.__name__} with lamda: {lamda} and padding: {padding}.")
        self.lamda = lamda
        self.padding = padding
        self._eps = eps
        self.symmetric = symmetric

    def forward(
            self, x_out: Tensor, x_tf_out: Tensor, mask: Tensor = None
    ) -> Tensor:
        if mask is not None:
            x_out *= mask
            x_tf_out *= mask
        T_side_dense = self.padding * 2 + 1
        if self.padding > 0:
            p_i_j = compute_joint_2D(x_out, x_tf_out, symmetric=self.symmetric, padding=self.padding)
        elif self.padding == 0:
            p_i_j = compute_joint_2D_with_padding_zeros(x_out, x_tf_out, symmetric=self.symmetric)
        else:
            raise ValueError(self.padding)
        self._p_i_j = p_i_j[0][0]

        # T x T x k x k
        p_i_mat = p_i_j.sum(dim=2, keepdim=True)
        p_j_mat = p_i_j.sum(dim=3, keepdim=True)

        # maximise information
        loss = -p_i_j * (
                torch.log(p_i_j + self._eps)
                - self.lamda * torch.log(p_i_mat + self._eps)
                - self.lamda * torch.log(p_j_mat + self._eps)
        )

        return loss.sum() / (T_side_dense * T_side_dense)

    def get_joint_matrix(self):
        if not hasattr(self, "_p_i_j"):
            raise RuntimeError()
        return self._p_i_j.detach().cpu().numpy()


class IIDSegmentationSmallPathLoss(IIDSegmentationLoss):

    def __init__(self, lamda=1.0, padding=7, eps: float = sys.float_info.epsilon, patch_size=32) -> None:
        super().__init__(lamda, padding, eps)
        self._patch_size = _pair(patch_size)
        self._step_size = _pair(patch_size // 2)

    def __call__(self, x_out: Tensor, x_tf_out: Tensor, mask: Tensor = None):
        assert x_out.shape == x_tf_out.shape, (x_out.shape, x_tf_out.shape)
        if mask is None:
            iic_patch_list = [super(IIDSegmentationSmallPathLoss, self).__call__(x, y) for x, y in zip(
                patch_generator(x_out, self._patch_size, self._step_size),
                patch_generator(x_tf_out, self._patch_size, self._step_size)
            )]
        else:
            iic_patch_list = [super(IIDSegmentationSmallPathLoss, self).__call__(x, y, m) for x, y, m in zip(
                patch_generator(x_out, self._patch_size, self._step_size),
                patch_generator(x_tf_out, self._patch_size, self._step_size),
                patch_generator(mask, self._patch_size, self._step_size)
            )]
        if any([torch.isnan(x) for x in iic_patch_list]):
            raise RuntimeError(iic_patch_list)
        return average_iter(iic_patch_list)

    def __repr__(self):
        return f"{self.__class__.__name__} with patch_size={self._patch_size} and padding={self.padding}."


def compute_joint(x_out: Tensor, x_tf_out: Tensor, symmetric=True) -> Tensor:
    r"""
    return joint probability
    :param x_out: p1, simplex
    :param x_tf_out: p2, simplex
    :param symmetric
    :return: joint probability
    """
    # produces variable that requires grad (since args require grad)
    assert simplex(x_out), f"x_out not normalized."
    assert simplex(x_tf_out), f"x_tf_out not normalized."

    bn, k = x_out.shape
    assert x_tf_out.size()[0] == bn and x_tf_out.size()[1] == k

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k aggregated over one batch
    if symmetric:
        p_i_j = (p_i_j + p_i_j.t()) / 2.0  # symmetric
    p_i_j /= p_i_j.sum()  # normalise

    return p_i_j.contiguous()


def compute_joint_2D(x_out: Tensor, x_tf_out: Tensor, *, symmetric: bool = True, padding: int = 0):
    k = x_out.shape[1]
    x_out = x_out.swapaxes(0, 1).contiguous()
    x_tf_out = x_tf_out.swapaxes(0, 1).contiguous()
    p_i_j = F.conv2d(
        input=x_out,
        weight=x_tf_out, padding=(int(padding), int(padding))
    )
    p_i_j = p_i_j - p_i_j.min().detach() + 1e-8

    # T x T x k x k
    p_i_j = p_i_j.permute(2, 3, 0, 1)
    p_i_j /= p_i_j.sum(dim=[2, 3], keepdim=True)  # norm

    # symmetrise, transpose the k x k part
    if symmetric:
        p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0
    p_i_j /= p_i_j.sum()  # norm
    return p_i_j.contiguous()


def compute_joint_2D_with_padding_zeros(x_out: Tensor, x_tf_out: Tensor, *, symmetric: bool = True):
    k = x_out.shape[1]
    x_out = x_out.swapaxes(0, 1).reshape(k, -1)
    N = x_out.shape[1]
    x_tf_out = x_tf_out.swapaxes(0, 1).reshape(k, -1)
    p_i_j = (x_out / math.sqrt(N)) @ (x_tf_out.t() / math.sqrt(N))
    # p_i_j = p_i_j - p_i_j.min().detach() + 1e-8

    # T x T x k x k
    # p_i_j /= p_i_j.sum()

    # symmetrise, transpose the k x k part
    if symmetric:
        p_i_j = (p_i_j + p_i_j.t()) / 2.0
    p_i_j = p_i_j.view(1, 1, k, k)
    return p_i_j.contiguous()


def patch_generator(feature_map, patch_size=(32, 32), step_size=(16, 16)):
    b, c, h, w = feature_map.shape
    hs = np.arange(0, h - patch_size[0], step_size[0])
    hs = np.append(hs, max(h - patch_size[0], 0))
    ws = np.arange(0, w - patch_size[1], step_size[1])
    ws = np.append(ws, max(w - patch_size[1], 0))
    for _h in hs:
        for _w in ws:
            yield feature_map[:, :, _h:min(_h + patch_size[0], h), _w:min(_w + patch_size[1], w)]


def imsat_loss(prediction: Tensor, lamda: float = 1.0):
    """
    this loss takes the input as both classification and segmentation
    """
    pred = prediction.moveaxis(0, 1).reshape(prediction.shape[1], -1)
    margin = pred.mean(1, keepdims=True)

    mi = -entropy_criterion(pred.t()).mean() + entropy_criterion(margin.t()).mean() * lamda

    return -mi


def imsat_with_entropy(prediction: Tensor, ):
    """
    this loss takes the input as both classification and segmentation
    """
    pred = prediction.moveaxis(0, 1).reshape(prediction.shape[1], -1)
    margin = pred.mean(1, keepdims=True)

    marginal = entropy_criterion(margin.t()).mean()
    conditional = entropy_criterion(pred.t()).mean()

    return marginal, conditional
