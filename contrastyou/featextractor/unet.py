from typing import List, Union

import torch
from torch import nn

from contrastyou.arch import UNet


# feature extractor
class _FeatureCollector:
    def __call__(self, _, input, result):
        self.feature = result

    def clear(self):
        pass


class _FeatureCollectorWithIndex:

    def __init__(self) -> None:
        self.__n = 0
        self.feature = dict()

    def __call__(self, _, input, result):
        self.feature[self.__n] = result
        self.__n += 1
        if self.__n >= 10:
            raise RuntimeError(f"You may forget to call clear as this hook "
                               f"has registered data from {self.__n} forward passes.")

    def clear(self):
        self.__n = 0
        del self.feature
        self.feature = dict()


class FeatureExtractor(nn.Module):

    def __init__(self, net: UNet, feature_names: Union[List[str], str]) -> None:
        super().__init__()
        self._net = net
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names
        for f in self._feature_names:
            assert f in UNet().component_names, f

    def __enter__(self):
        self._feature_exactors = {}
        self._hook_handlers = {}
        from semi_seg._utils import get_model
        net = get_model(self._net)
        for f in self._feature_names:
            extractor = self._create_collector()
            handler = getattr(net, f).register_forward_hook(extractor)
            self._feature_exactors[f] = extractor
            self._hook_handlers[f] = handler
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self._hook_handlers.items():
            v.remove()
        del self._feature_exactors, self._hook_handlers

    def __getitem__(self, item):
        if item in self._feature_exactors:
            return self._feature_exactors[item].feature
        return super().__getitem__(item)

    def get_feature_from_num(self, num):
        feature = self._feature_names[num]
        return self[feature]

    def __iter__(self):
        for k, v in self._feature_exactors.items():
            yield v.feature

    def _create_collector(self):
        return _FeatureCollector()

    def clear(self):
        pass


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
            yield torch.cat(list(v.feature.values()), dim=0)
