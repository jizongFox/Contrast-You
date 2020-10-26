from itertools import repeat
from typing import List, Union

from torch import nn, Tensor
from torch._six import container_abcs

from contrastyou.arch import UNet
from contrastyou.losses.iic_loss import IIDLoss as _IIDLoss, IIDSegmentationSmallPathLoss
from contrastyou.losses.pica_loss import PUILoss, PUISegLoss
from contrastyou.losses.wrappers import LossWrapperBase
from contrastyou.projectors.heads import LocalClusterHead as _LocalClusterHead, ClusterHead as _EncoderClusterHead, \
    ProjectionHead as _ProjectionHead, LocalProjectionHead as _LocalProjectionHead
from contrastyou.projectors.wrappers import ProjectorWrapperBase, CombineWrapperBase


def get_model(model):
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
        return model.module
    elif isinstance(model, nn.Module):
        return model
    raise TypeError(type(model))


class _num_class_mixin:
    _model: nn.Module

    @property
    def num_classes(self):
        return get_model(self._model).num_classes


class IIDLoss(_IIDLoss):

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        return super().forward(x_out, x_tf_out)[0]


def _filter_encodernames(feature_list):
    encoder_list = UNet().encoder_names
    return list(filter(lambda x: x in encoder_list, feature_list))


def _filter_decodernames(feature_list):
    decoder_list = UNet().decoder_names
    return list(filter(lambda x: x in decoder_list, feature_list))


def _nlist(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            assert len(x) == n, (len(x), n)
            return x
        return list(repeat(x, n))

    return parse


class FeatureExtractor(nn.Module):
    class _FeatureExtractor:
        def __call__(self, _, input, result):
            self.feature = result

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
        net = get_model(self._net)
        for f in self._feature_names:
            extractor = self._FeatureExtractor()
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


class _LocalClusterWrappaer(ProjectorWrapperBase):
    def __init__(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]] = "linear",
        num_subheads: Union[int, List[int]] = 5,
        num_clusters: Union[int, List[int]] = 10,
        normalize: Union[bool, List[bool]] = False
    ) -> None:
        super(_LocalClusterWrappaer, self).__init__()
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names

        n_pair = _nlist(len(feature_names))

        self._head_types = n_pair(head_types)
        self._num_subheads = n_pair(num_subheads)
        self._num_clusters = n_pair(num_clusters)
        self._normalize = n_pair(normalize)

        for f, h, c, s, n in zip(self._feature_names, self._head_types, self._num_clusters, self._num_subheads,
                                 self._normalize):
            self._projectors[f] = self._create_clusterheads(
                input_dim=UNet.dimension_dict[f],
                head_type=h,
                num_clusters=c,
                num_subheads=s,
                normalize=n
            )

    @staticmethod
    def _create_clusterheads(*args, **kwargs):
        return _LocalClusterHead(*args, **kwargs)


class _EncoderClusterWrapper(_LocalClusterWrappaer):
    @staticmethod
    def _create_clusterheads(*args, **kwargs):
        return _EncoderClusterHead(*args, **kwargs)


class ClusterProjectorWrapper(CombineWrapperBase):

    def __init__(self):
        super().__init__()
        self._encoder_names = []
        self._decoder_names = []

    def init_encoder(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]] = "linear",
        num_subheads: Union[int, List[int]] = 5,
        num_clusters: Union[int, List[int]] = 10,
        normalize: Union[bool, List[bool]] = False
    ):
        self._encoder_names = _filter_encodernames(feature_names)
        encoder_projectors = _EncoderClusterWrapper(
            self._encoder_names, head_types, num_subheads,
            num_clusters, normalize)
        self._projector_list.append(encoder_projectors)

    def init_decoder(self,
                     feature_names: Union[str, List[str]],
                     head_types: Union[str, List[str]] = "linear",
                     num_subheads: Union[int, List[int]] = 5,
                     num_clusters: Union[int, List[int]] = 10,
                     normalize: Union[bool, List[bool]] = False
                     ):
        self._decoder_names = _filter_decodernames(feature_names)
        decoder_projectors = _LocalClusterWrappaer(
            self._decoder_names, head_types, num_subheads,
            num_clusters, normalize)
        self._projector_list.append(decoder_projectors)

    @property
    def feature_names(self):
        return self._encoder_names + self._decoder_names


class _ContrastiveEncodeProjectorWrapper(ProjectorWrapperBase):

    def __init__(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]],
    ):
        super().__init__()
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names
        n = len(self._feature_names)
        n_pair = _nlist(n)
        self._head_types = n_pair(head_types)
        for f, h in zip(self._feature_names, self._head_types):
            self._projectors[f] = self._create_head(
                input_dim=UNet.dimension_dict[f],
                head_type=h,
                output_dim=256,
            )

    @staticmethod
    def _create_head(input_dim, output_dim, head_type):
        return _ProjectionHead(input_dim=input_dim, output_dim=output_dim, head_type=head_type)


class _ContrastiveDecoderProjectorWrapper(_ContrastiveEncodeProjectorWrapper):
    @staticmethod
    def _create_head(input_dim, output_dim, head_type):
        return _LocalProjectionHead(input_dim=input_dim, head_type=head_type, output_size=(2, 2))


class ContrastiveProjectorWrapper(CombineWrapperBase):

    def __init__(self):
        super().__init__()
        self._encoder_names = []
        self._decoder_names = []

    def init_encoder(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]] = "mlp",
    ):
        self._encoder_names = _filter_encodernames(feature_names)
        encoder_projectors = _ContrastiveEncodeProjectorWrapper(
            feature_names=self._encoder_names, head_types=head_types)
        self._projector_list.append(encoder_projectors)

    def init_decoder(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]] = "mlp",
    ):
        self._decoder_names = _filter_decodernames(feature_names)
        decoder_projectors = _ContrastiveDecoderProjectorWrapper(
            self._decoder_names, head_types)
        self._projector_list.append(decoder_projectors)

    @property
    def feature_names(self):
        return self._encoder_names + self._decoder_names


class IICLossWrapper(LossWrapperBase):

    def __init__(self,
                 feature_names: Union[str, List[str]],
                 paddings: Union[int, List[int]],
                 patch_sizes: Union[int, List[int]]) -> None:
        super().__init__()
        self._encoder_features = _filter_encodernames(feature_names)
        self._decoder_features = _filter_decodernames(feature_names)
        assert len(feature_names) == len(self._encoder_features) + len(self._decoder_features)

        if len(self._encoder_features) > 0:
            for f in self._encoder_features:
                self._LossModuleDict[f] = IIDLoss()
        if len(self._decoder_features) > 0:
            paddings = _nlist(len(self._decoder_features))(paddings)
            patch_sizes = _nlist(len(self._decoder_features))(patch_sizes)
            for f, p, size in zip(self._decoder_features, paddings, patch_sizes):
                self._LossModuleDict[f] = IIDSegmentationSmallPathLoss(padding=p, patch_size=size)

    @property
    def feature_names(self):
        return self._encoder_features + self._decoder_features


class PICALossWrapper(LossWrapperBase):

    def __init__(self,
                 feature_names: Union[str, List[str]],
                 paddings: Union[int, List[int]]) -> None:
        super().__init__()
        self._encoder_features = _filter_encodernames(feature_names)
        self._decoder_features = _filter_decodernames(feature_names)
        assert len(feature_names) == len(self._encoder_features) + len(self._decoder_features)

        if len(self._encoder_features) > 0:
            for f in self._encoder_features:
                self._LossModuleDict[f] = PUILoss()
        if len(self._decoder_features) > 0:
            paddings = _nlist(len(self._decoder_features))(paddings)
            for f, p in zip(self._decoder_features, paddings):
                self._LossModuleDict[f] = PUISegLoss(padding=p)

    @property
    def feature_names(self):
        return self._encoder_features + self._decoder_features
