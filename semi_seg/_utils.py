from typing import List, Union

from torch import nn

from contrastyou.arch import UNet


class FeatureExtractor(nn.Module):

    def __init__(self, net: UNet, feature_names: Union[List[str], str]) -> None:
        super().__init__()
        self._net = net
        if isinstance(feature_names, str):
            feature_names = (feature_names,)
        self._feature_names = feature_names
        for f in self._feature_names:
            assert f in UNet().component_names, f

    def __enter__(self):
        self._feature_exactors = {}
        self._hook_handlers = {}
        for f in self._feature_names:
            extractor = _FeatureExtractor()
            handler = getattr(self._net, f).register_forward_hook(extractor)
            self._feature_exactors[f] = extractor
            self._hook_handlers[f] = handler
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._feature_exactors = {}
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


class _FeatureExtractor:
    def __call__(self, _, input, result):
        self.feature = result
