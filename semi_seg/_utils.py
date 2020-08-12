from itertools import repeat
from typing import List, Union

from torch import nn
from torch._six import container_abcs

from contrastyou.arch import UNet
from contrastyou.trainer._utils import LocalClusterHead as _LocalClusterHead


def _nlist(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            assert len(x) == n, (len(x), n)
            return x
        return list(repeat(x, n))

    return parse


class _FeatureExtractor:
    def __call__(self, _, input, result):
        self.feature = result


class FeatureExtractor(nn.Module):

    def __init__(self, net: UNet, feature_names: Union[List[str], str]) -> None:
        super().__init__()
        self._net = net
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names
        for f in self._feature_names:
            assert f in UNet().component_names, f
        self._feature_exactors = {}

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

    def __iter__(self):
        for k, v in self._feature_exactors.items():
            yield v.feature


class LocalClusterWrappaer(nn.Module):
    def __init__(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]] = "linear",
        num_subheads: Union[int, List[int]] = 5,
        num_clusters: Union[int, List[int]] = 10,
    ) -> None:
        super(LocalClusterWrappaer, self).__init__()
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names

        n_pair = _nlist(len(self))

        self._head_types = n_pair(head_types)
        self._num_subheads = n_pair(num_subheads)
        self._num_clusters = n_pair(num_clusters)

        self._clusters = nn.ModuleDict()

        for f, h, c, s in zip(self._feature_names, self._head_types, self._num_clusters, self._num_subheads):
            self._clusters[f] = _LocalClusterHead(UNet.dimension_dict[f], h, c, s)

    def __len__(self):
        return len(self._feature_names)

    def __iter__(self):
        for k, v in self._clusters.items():
            yield v

    def __getitem__(self, item):
        if item in self._clusters.keys():
            return self._clusters[item]
        return super().__getitem__(item)
