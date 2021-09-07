import numpy as np
import torch
from contrastyou.utils.general import simplex, assert_list
from loguru import logger
from torch import Tensor
from torch import nn
from typing import Optional, OrderedDict
from typing import TypeVar, List, Dict, Union

__all__ = ["Entropy", "KL_div", "JSD_div"]

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)


def _check_reduction_params(reduction):
    assert reduction in (
        "mean",
        "sum",
        "none",
    ), "reduction should be in ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``, given {}".format(
        reduction
    )


class Entropy(nn.Module):
    r"""General Entropy interface

    the definition of Entropy is - \sum p(xi) log (p(xi))

    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16):
        super().__init__()
        _check_reduction_params(reduction)
        self._eps = eps
        self._reduction = reduction

    def forward(self, input_: Tensor) -> Tensor:
        assert input_.shape.__len__() >= 2
        b, _, *s = input_.shape
        assert simplex(input_), f"Entropy input should be a simplex"
        e = input_ * (input_ + self._eps).log()
        e = -1.0 * e.sum(1)
        assert e.shape == torch.Size([b, *s])
        if self._reduction == "mean":
            return e.mean()
        elif self._reduction == "sum":
            return e.sum()
        else:
            return e


class KL_div(nn.Module):
    """
    KL(p,q)= -\sum p(x) * log(q(x)/p(x))
    where p, q are distributions
    p is usually the fixed one like one hot coding
    p is the target and q is the distribution to get approached.

    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16, weight: Union[List[float], Tensor] = None):
        super().__init__()
        _check_reduction_params(reduction)
        self._eps = eps
        self._reduction = reduction
        self._weight: Optional[Tensor] = weight
        if weight is not None:
            assert isinstance(weight, (list, Tensor)), type(weight)
            if isinstance(weight, list):
                assert assert_list(lambda x: isinstance(x, (int, float)), weight)
                self._weight = torch.Tensor(weight).float()
            else:
                self._weight = weight.float()
            # normalize weight:
            self._weight = self._weight / self._weight.sum() * len(self._weight)
        logger.trace(
            f"Initialized {self.__class__.__name__} with weight={self._weight} and reduction={self._reduction}.")

    def forward(self, prob: Tensor, target: Tensor, **kwargs) -> Tensor:
        b, c, *hwd = target.shape
        kl = (-target * torch.log((prob + self._eps) / (target + self._eps)))
        if self._weight is not None:
            assert len(self._weight) == c
            weight = self._weight.expand(b, *hwd, -1).transpose(-1, 1).detach()
            kl *= weight.to(kl.device)
        kl = kl.sum(1)
        if self._reduction == "mean":
            return kl.mean()
        elif self._reduction == "sum":
            return kl.sum()
        else:
            return kl

    def __repr__(self):
        return f"{self.__class__.__name__}\n, weight={self._weight}"

    def state_dict(self, *args, **kwargs):
        save_dict = super().state_dict(*args, **kwargs)
        # save_dict["weight"] = self._weight
        # save_dict["reduction"] = self._reduction
        return save_dict

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], OrderedDict[str, Tensor]], *args, **kwargs):
        super(KL_div, self).load_state_dict(state_dict, **kwargs)
        # self._reduction = state_dict["reduction"]
        # self._weight = state_dict["weight"]


class JSD_div(nn.Module):
    """
    general JS divergence interface
    :<math>{\rm JSD}_{\pi_1, \ldots, \pi_n}(P_1, P_2, \ldots, P_n) = H\left(\sum_{i=1}^n \pi_i P_i\right) - \sum_{i=1}^n \pi_i H(P_i)</math>


    reduction (string, optional): Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16):
        super().__init__()
        _check_reduction_params(reduction)
        self._reduction = reduction
        self._eps = eps
        self._entropy_criterion = Entropy(reduction=reduction, eps=eps)

    def forward(self, *input_: Tensor) -> Tensor:
        assert assert_list(
            lambda x: simplex(x), input_
        ), f"input tensor should be a list of simplex."
        assert assert_list(
            lambda x: x.shape == input_[0].shape, input_
        ), "input tensor should have the same dimension"
        mean_prob = sum(input_) / len(input_)
        f_term = self._entropy_criterion(mean_prob)
        mean_entropy: Tensor = sum(
            list(map(lambda x: self._entropy_criterion(x), input_))
        ) / len(input_)
        assert f_term.shape == mean_entropy.shape
        return f_term - mean_entropy
