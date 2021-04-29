from contextlib import contextmanager
from typing import Union

from loguru import logger
from torch import nn, Tensor

from semi_seg.arch.hook import FeatureExtractor


class unl_extractor:
    def __init__(self, features: FeatureExtractor, n_uls: int) -> None:
        super().__init__()
        self._features = features
        self._n_uls = n_uls

    def __iter__(self):
        for feature in self._features:
            assert len(feature) >= self._n_uls, (len(feature), self._n_uls)
            yield feature[len(feature) - self._n_uls:]


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


class __AssertWithUnLabeledData:

    def _assertion(self):
        from .base import FineTuneEpocher
        assert not isinstance(self, FineTuneEpocher)
        logger.debug("{} using unlabeled data checked!", self.__class__.__name__)


class __AssertOnlyWithLabeledData:

    def _assertion(self):
        from .base import FineTuneEpocher
        assert isinstance(self, FineTuneEpocher)
        logger.debug("{} using only labeled data checked!", self.__class__.__name__)
