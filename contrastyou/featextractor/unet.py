from contextlib import contextmanager
from typing import List, Union

import torch
from torch import nn

from contrastyou.arch import UNet


# feature extractor
class _FeatureCollector:

    def __init__(self) -> None:
        super().__init__()
        self._enable = False
        self.feature = None

    def __call__(self, _, input, result):
        if self._enable:
            self.feature = result

    def clear(self):
        pass

    def set_enable(self, enable=True):
        self._enable = enable

    @property
    def enable(self):
        return self._enable


class _FeatureCollectorWithIndex:

    def __init__(self) -> None:
        self.__n = 0
        self.feature = dict()
        self._enable = False

    def __call__(self, _, input, result):
        if self._enable:
            self.feature[self.__n] = result
            self.__n += 1
            if self.__n >= 10:
                raise RuntimeError(f"You may forget to call clear as this hook "
                                   f"has registered data from {self.__n} forward passes.")

    def clear(self):
        self.__n = 0
        del self.feature
        self.feature = dict()

    def set_enable(self, enable=True):
        self._enable = enable

    @property
    def enable(self):
        return self._enable


class FeatureExtractor(nn.Module):

    def __init__(self, net: UNet, feature_names: Union[List[str], str]) -> None:
        super().__init__()
        self._net = net
        self.feature_importance = None
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names
        for f in self._feature_names:
            assert f in net.component_names, f
        self.bind()

    def bind(self):
        self._feature_exactors = {}
        self._hook_handlers = {}
        from semi_seg.utils import get_model
        net = get_model(self._net)
        for i, f in enumerate(self._feature_names):
            extractor = self._create_collector()
            handler = getattr(net, f).register_forward_hook(extractor)
            self._feature_exactors[str(i) + "_" + f] = extractor
            self._hook_handlers[str(i) + "_" + f] = handler

    def remove(self):
        for k, v in self._hook_handlers.items():
            v.remove()
        del self._feature_exactors, self._hook_handlers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

    def __getitem__(self, item):
        if item in self._feature_exactors:
            return self._feature_exactors[item].feature
        return super().__getitem__(item)

    def __iter__(self):
        for k, v in self._feature_exactors.items():
            yield v.feature

    def _create_collector(self):
        return _FeatureCollector()

    def clear(self):
        pass

    @property
    def feature_names(self):
        return self._feature_names

    def enable_register(self):
        return enable_register(self)

    def disable_register(self):
        return disable_register(self)


class FeatureExtractorWithIndex(FeatureExtractor):
    """
    This module enhance the FeatureExtractor to provide multiple forward record with `clear` method
    """

    def _create_collector(self):
        return _FeatureCollectorWithIndex()

    def clear(self):
        for k, v in self._feature_exactors.items():
            v.clear()

    def __iter__(self):
        for k, v in self._feature_exactors.items():
            extracted_list = list(v.feature.values())
            if len(extracted_list) > 0:
                yield torch.cat(list(v.feature.values()), dim=0)
            else:
                raise RuntimeError(f"{self.__class__.__name__} found no element been registered. "
                                   f"Call `with enable_register(): within a context manager should be perform first.`")


@contextmanager
def enable_register(feature_extractor: FeatureExtractor):
    if not hasattr(feature_extractor, "_feature_exactors"):
        raise RuntimeError(f"{feature_extractor.__class__.__name__} should be put in with first")
    previous_state = list(feature_extractor._feature_exactors.values())[0].enable  # noqa
    for v in feature_extractor._feature_exactors.values():  # noqa
        v.set_enable(enable=True)
    yield feature_extractor
    for v in feature_extractor._feature_exactors.values():  # noqa
        v.set_enable(enable=previous_state)


@contextmanager
def disable_register(feature_extractor: FeatureExtractor):
    if not hasattr(feature_extractor, "_feature_exactors"):
        raise RuntimeError(f"{feature_extractor.__class__.__name__} should be put in with first")
    previous_state = list(feature_extractor._feature_exactors.values())[0].enable  # noqa
    for v in feature_extractor._feature_exactors.values():  # noqa
        v.set_enable(enable=False)
    yield feature_extractor
    for v in feature_extractor._feature_exactors.values():  # noqa
        v.set_enable(enable=previous_state)
