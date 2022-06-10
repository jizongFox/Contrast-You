from collections import OrderedDict
from contextlib import contextmanager, ExitStack
from typing import Union, List, Iterator

import torch
from loguru import logger
from torch.nn import Parameter, Module

from contrastyou.arch._base import _Network

__all__ = ["get_requires_grad", "get_bn_track", "SingleFeatureExtractor", "FeatureExtractor"]


def get_requires_grad(input_: Union[Parameter, Module]) -> bool:
    """this just check the first element of the module, thus causing errors if inconsistency occurs"""
    assert isinstance(input_, (Parameter, Module)), type(input_)
    if isinstance(input_, Module):
        return next(input_.parameters()).requires_grad
    return input_.requires_grad


def get_bn_track(input_: Module) -> bool:
    """this just check the first bn submodule of the module, thus causing errors if inconsistency occurs"""
    if hasattr(input_, "track_running_stats"):
        return input_.track_running_stats  # type: bool
    for m in input_.modules():
        if hasattr(m, "track_running_stats"):
            return m.track_running_stats
    raise RuntimeError(f"BN module not found in {input_}")


class _FeatureCollector:

    def __init__(self, max_limit=5) -> None:
        self.__n = 0
        self.feature: OrderedDict = OrderedDict()
        self._enable = False
        self.__max = max_limit

    def __call__(self, _, input_, result):
        if self._enable:
            self.feature[self.__n] = result
            self.__n += 1
            if self.__n >= self.__max:
                raise RuntimeError(f"You may forget to call clear as this hook "
                                   f"has registered data from {self.__n} forward passes.")
        return

    def clear(self):
        self.__n = 0
        del self.feature
        self.feature: OrderedDict = OrderedDict()

    def set_enable(self, enable=True):
        self._enable = enable

    @property
    def enable(self):
        return self._enable


class SingleFeatureExtractor:

    def __init__(self, model: _Network, feature_name: str) -> None:
        self._model = model
        self._feature_name = feature_name
        assert self._feature_name in model.arch_elements, self._feature_name
        self._feature_extractor: _FeatureCollector = None  # type: ignore # noqa
        self._hook_handler = None
        self.__bind_done__ = False

    def bind(self):
        logger.opt(depth=3).trace(f"Binding {self.__class__.__name__}@{self._feature_name}")
        extractor = _FeatureCollector()
        handler = self._model.get_module(self._feature_name).register_forward_hook(extractor)
        self._feature_extractor = extractor
        self._hook_handler = handler
        self.__bind_done__ = True

    def remove(self):
        logger.opt(depth=3).trace(f"Remove {self.__class__.__name__}@{self._feature_name}")
        self._hook_handler.remove()
        self.__bind_done__ = False

    def __enter__(self, ):
        self.bind()
        return self

    def __exit__(self, *args, **kwargs):
        self.remove()

    def clear(self):
        self._feature_extractor.clear()

    def feature(self):
        collected_feature_dict = self._feature_extractor.feature
        if len(collected_feature_dict) > 0:
            return torch.cat(list(collected_feature_dict.values()), dim=0)
        raise RuntimeError("no feature has been recorded.")

    def set_enable(self, enable=True):
        self._feature_extractor.set_enable(enable=enable)

    @contextmanager
    def enable_register(self, enable=True):
        prev_state = self._feature_extractor.enable
        logger.opt(depth=3).trace(f"{'enable' if enable else 'disable'} recording")
        self.set_enable(enable)
        yield
        logger.opt(depth=3).trace("restore previous recording status")
        self.set_enable(prev_state)


class FeatureExtractor:

    def __init__(self, model: _Network, feature_names: Union[str, List[str]]):
        self._feature_names = (feature_names,) if isinstance(feature_names, str) else feature_names
        self._extractor_list = [SingleFeatureExtractor(model, f) for f in self._feature_names]

    def bind(self):
        for e in self._extractor_list:
            e.bind()

    def remove(self):
        for e in self._extractor_list:
            e.remove()

    def __enter__(self):
        self.bind()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

    def set_enable(self, enable=True):
        for e in self._extractor_list:
            e.set_enable(enable)

    @contextmanager
    def enable_register(self, enable=True):
        with ExitStack() as stack:
            for e in self._extractor_list:
                stack.enter_context(e.enable_register(enable=enable))
            yield

    def clear(self):
        for e in self._extractor_list:
            e.clear()

    def __iter__(self):
        for e in self._extractor_list:
            yield e.feature()

    def features(self) -> Iterator:
        return iter(self)

    def named_features(self) -> Iterator:
        for name, feature in zip(self._feature_names, self.features()):
            yield name, feature
