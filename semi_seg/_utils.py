from itertools import repeat
from typing import List, Union

from torch import nn, Tensor
from torch._six import container_abcs

from contrastyou.arch import UNet
from contrastyou.losses.iic_loss import IIDLoss as _IIDLoss, IIDSegmentationSmallPathLoss
from contrastyou.losses.pica_loss import PUILoss, PUISegLoss
from contrastyou.losses.wrappers import LossWrapperBase
from contrastyou.projectors.heads import DenseClusterHead as _LocalClusterHead, ClusterHead as _EncoderClusterHead, \
    ProjectionHead as _ProjectionHead, DenseProjectionHead as _LocalProjectionHead
from contrastyou.projectors.wrappers import ProjectorWrapperBase, CombineWrapperBase


def get_model(model):
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
        return model.module
    elif isinstance(model, nn.Module):
        return model
    raise TypeError(type(model))


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


class _num_class_mixin:
    _model: nn.Module

    @property
    def num_classes(self):
        return get_model(self._model).num_classes


# encoder contrastive projector
class _ContrastiveEncodeProjectorWrapper(ProjectorWrapperBase):

    def __init__(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]],
        normalize: Union[bool, List[bool]] = True,
        pooling_method: Union[str, List[str]] = None
    ):
        super().__init__()
        if isinstance(pooling_method, str):
            assert pooling_method in ("adaptive_avg", "adaptive_max"), pooling_method
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names
        n = len(self._feature_names)
        n_pair = _nlist(n)
        self._head_types = n_pair(head_types)
        self._normalize = n_pair(normalize)
        self._pooling_method = n_pair(pooling_method)
        for i, (f, h, n, p) in enumerate(
            zip(self._feature_names, self._head_types, self._normalize, self._pooling_method)):
            self._projectors[str(i) + "_" + f] = self._create_head(
                input_dim=UNet.dimension_dict[f],
                head_type=h,
                output_dim=256,
                normalize=n,
                pooling_method=p
            )

    def _create_head(self, input_dim, output_dim, head_type, normalize, pooling_method):
        if pooling_method is not None:
            return _ProjectionHead(input_dim=input_dim, output_dim=output_dim, head_type=head_type, normalize=normalize,
                                   pooling_name=pooling_method)
        # if no pooling method is provided, output dense prediction of 14\times14, with 1x1 conv.
        return _LocalProjectionHead(input_dim=input_dim, head_type=head_type, output_size=None, normalize=normalize,
                                    pooling_name="none")


# decoder contrastive projector
class _ContrastiveDecoderProjectorWrapper(_ContrastiveEncodeProjectorWrapper):

    def __init__(self, feature_names: Union[str, List[str]], head_types: Union[str, List[str]],
                 normalize: Union[bool, List[bool]] = True, output_size=(2, 2), pooling_method: str = None):
        self._output_size = output_size

        super().__init__(feature_names, head_types, normalize, pooling_method)

    def _create_head(self, input_dim, output_dim, head_type, normalize, pooling_method):
        return _LocalProjectionHead(input_dim=input_dim, head_type=head_type, output_size=self._output_size,
                                    normalize=normalize, pooling_name=pooling_method)


# encoder and decoder contrastive projector
class ContrastiveProjectorWrapper(CombineWrapperBase):

    def __init__(self):
        super().__init__()
        self._encoder_names = []
        self._decoder_names = []

    def init_encoder(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]] = "mlp",
        normalize: Union[bool, List[bool]] = True,
        pool_method: str = None
    ):
        if isinstance(pool_method, str):
            pool_method = [pool_method, ]
        if isinstance(pool_method, (list, tuple)):
            for p in pool_method:
                assert p in ("adaptive_avg", "adaptive_max", "identical"), pool_method
        pool_method = [p if p != "identical" else None for p in pool_method]

        self._encoder_names = _filter_encodernames(feature_names)
        encoder_projectors = _ContrastiveEncodeProjectorWrapper(
            feature_names=self._encoder_names, head_types=head_types, normalize=normalize, pooling_method=pool_method)
        self._projector_list.append(encoder_projectors)

    def init_decoder(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]] = "mlp",
        normalize: Union[bool, List[bool]] = True,
        output_size=(2, 2),
        pool_method: str = None
    ):
        assert pool_method in ("adaptive_avg", "adaptive_max"), pool_method
        self._decoder_names = _filter_decodernames(feature_names)
        decoder_projectors = _ContrastiveDecoderProjectorWrapper(
            self._decoder_names, head_types, normalize=normalize, output_size=output_size, pooling_method=pool_method)
        self._projector_list.append(decoder_projectors)

    @property
    def feature_names(self):
        return self._encoder_names + self._decoder_names


# decoder IIC projectors
class _LocalClusterWrapper(ProjectorWrapperBase):
    def __init__(
        self,
        feature_names: Union[str, List[str]],
        head_types: Union[str, List[str]] = "linear",
        num_subheads: Union[int, List[int]] = 5,
        num_clusters: Union[int, List[int]] = 10,
        normalize: Union[bool, List[bool]] = False,
        temperature: Union[float, List[float]] = 1.0,
    ) -> None:
        super(_LocalClusterWrapper, self).__init__()
        if isinstance(feature_names, str):
            feature_names = [feature_names, ]
        self._feature_names = feature_names

        n_pair = _nlist(len(feature_names))

        self._head_types = n_pair(head_types)
        self._num_subheads = n_pair(num_subheads)
        self._num_clusters = n_pair(num_clusters)
        self._normalize = n_pair(normalize)
        self._temperature = n_pair(temperature)

        for f, h, c, s, n, t in zip(self._feature_names, self._head_types, self._num_clusters, self._num_subheads,
                                    self._normalize, self._temperature):
            self._projectors[f] = self._create_clusterheads(
                input_dim=UNet.dimension_dict[f],
                head_type=h,
                num_clusters=c,
                num_subheads=s,
                normalize=n,
                T=t
            )

    @staticmethod
    def _create_clusterheads(*args, **kwargs):
        return _LocalClusterHead(*args, **kwargs)


# encoder IIC projectors
class _EncoderClusterWrapper(_LocalClusterWrapper):
    @staticmethod
    def _create_clusterheads(*args, **kwargs):
        return _EncoderClusterHead(*args, **kwargs)


# encoder and decoder projectors for IIC
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
        normalize: Union[bool, List[bool]] = False,
        temperature: Union[float, List[float]] = 1.0
    ):
        self._encoder_names = _filter_encodernames(feature_names)
        encoder_projectors = _EncoderClusterWrapper(
            self._encoder_names, head_types, num_subheads,
            num_clusters, normalize, temperature)
        self._projector_list.append(encoder_projectors)

    def init_decoder(self,
                     feature_names: Union[str, List[str]],
                     head_types: Union[str, List[str]] = "linear",
                     num_subheads: Union[int, List[int]] = 5,
                     num_clusters: Union[int, List[int]] = 10,
                     normalize: Union[bool, List[bool]] = False,
                     temperature: Union[float, List[float]] = 1.0
                     ):
        self._decoder_names = _filter_decodernames(feature_names)
        decoder_projectors = _LocalClusterWrapper(
            self._decoder_names, head_types, num_subheads,
            num_clusters, normalize, temperature)
        self._projector_list.append(decoder_projectors)

    @property
    def feature_names(self):
        return self._encoder_names + self._decoder_names


# loss function
class IIDLoss(_IIDLoss):

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        return super().forward(x_out, x_tf_out)[0]


# IIC loss for encoder and decoder
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


# PICA loss for encoder and decoder
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
