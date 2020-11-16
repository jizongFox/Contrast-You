from contextlib import contextmanager
from typing import Union

import torch
from torch import Tensor, nn

from semi_seg._utils import FeatureExtractor


class unl_extractor:
    def __init__(self, features: FeatureExtractor, n_uls: int) -> None:
        super().__init__()
        self._features = features
        self._n_uls = n_uls

    def __iter__(self):
        for feature in self._features:
            assert len(feature) >= self._n_uls, (len(feature), self._n_uls)
            yield feature[len(feature) - self._n_uls:]


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


@contextmanager
def set_grad_tensor(tensor: Tensor, is_enable: bool):
    prev_flag = tensor.requires_grad
    tensor.requires_grad = is_enable
    yield
    tensor.requires_grad = prev_flag


@contextmanager
def set_grad_module(module: nn.Module, is_enable: bool):
    prev_flags = {k: v.requires_grad for k, v in module.named_parameters()}
    for k, v in module.named_parameters():
        v.requires_grad = is_enable
    yield
    for k, v in module.named_parameters():
        v.requires_grad = prev_flags[k]


def set_grad(tensor_or_module: Union[Tensor, nn.Module], is_enable):
    assert isinstance(tensor_or_module, (Tensor, nn.Module))
    if isinstance(tensor_or_module, Tensor):
        return set_grad_tensor(tensor_or_module, is_enable)
    return set_grad_module(tensor_or_module, is_enable)
