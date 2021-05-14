from typing import Iterator

import torch
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.meters2 import AverageValueMeter, MeterInterface
from torch import nn
from torch.nn import Parameter

from semi_seg.arch import UNet
from semi_seg.arch import get_channel_dim
from semi_seg.arch.hook import SingleFeatureExtractor

encoder_names, decoder_names = UNet.encoder_names, UNet.decoder_names


class MIEstimatorHook(nn.Module):
    def __init__(self, *, model: nn.Module, feature_name: str, weight: float = 1.0, num_clusters=20, num_subheads=5,
                 padding=None) -> None:
        super().__init__()
        assert feature_name in encoder_names + decoder_names, feature_name
        self._feature_name = feature_name
        self._extractor = SingleFeatureExtractor(model, feature_name=feature_name)  # noqa
        self._weight = weight
        input_dim = get_channel_dim(layer_name=feature_name)
        self._projector = self.init_projector(input_dim=input_dim, num_clusters=num_clusters, num_subheads=num_subheads)
        self._criterion = self.init_criterion(padding=padding)
        self._learnable_models = (self._projector,)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for m in self._learnable_models:
            yield from m.parameters(recurse=recurse)

    def init_projector(self, *, input_dim, num_clusters, num_subheads=5):
        self._projector = self.projector_class(input_dim=input_dim, num_clusters=num_clusters,
                                               num_subheads=num_subheads, head_type="linear", T=1, normalize=False)
        return self._projector

    def init_criterion(self, padding: int = None):
        if self._feature_name in encoder_names:
            return self._init_criterion()
        return self._init_dense_criterion(padding=padding or 1)

    def __call__(self):
        return _EpochHook(weight=self._weight, extractor=self._extractor, projector=self._projector,
                          criterion=self._criterion)

    def _init_dense_criterion(self, padding: int = 3):
        self._criterion = self.criterion_class(padding=padding)
        return self._criterion

    def _init_criterion(self):
        self._criterion = self.criterion_class()
        return self._criterion

    @property
    def projector_class(self):
        from contrastyou.projectors.heads import DenseClusterHead, ClusterHead
        if self._feature_name in encoder_names:
            return ClusterHead
        return DenseClusterHead

    @property
    def criterion_class(self):
        from contrastyou.losses.iic_loss import IIDLoss, IIDSegmentationLoss
        if self._feature_name in encoder_names:
            return IIDLoss
        return IIDSegmentationLoss

    def __enter__(self):
        self._extractor.bind()
        return self

    def __exit__(self, *args, **kwargs):
        self._extractor.remove()


class _EpochHook:
    def __init__(self, *, weight, extractor, projector, criterion):
        self._extractor = extractor
        self._extractor.bind()
        self._weight = weight
        self._projector = projector
        self._criterion = criterion
        self._epocher = None
        self.meters = MeterInterface()

    @staticmethod
    def configure_meters(meters):
        meters.register_meter("mi", AverageValueMeter())

    @property
    def epocher(self):
        if self._epocher is None:
            raise RuntimeError()
        return self._epocher

    @epocher.setter
    def epocher(self, epocher):
        self._epocher = epocher
        self.configure_meters(self.meters)

    def before_forward_pass(self, **kwargs):
        self._extractor.clear()
        self._extractor.set_enable(True)

    def after_forward_pass(self, **kwargs):
        self._extractor.set_enable(False)

    def before_regularization(self, **kwargs):
        pass

    def after_regularization(self, **kwargs):
        pass

    def __call__(self, *, affine_transformer, seed, **kwargs):
        losses = []
        feature_=self._extractor.feature()
        proj_feature, proj_tf_feature = torch.chunk(feature_, 2, dim=0)
        with FixRandomSeed(seed):
            proj_feature_tf = affine_transformer(proj_feature)

        prob1, prob2 = list(
            zip(*[torch.chunk(x, 2, 0) for x in self._projector(
                torch.cat([proj_feature_tf, proj_tf_feature], dim=0)
            )])
        )

        loss = sum([self._criterion(x1, x2)[0] for x1, x2 in zip(prob1, prob2)]) / len(prob1)
        losses.append(loss)
        self.meters["mi"].add(sum(losses).item())
        return sum(losses) * self._weight
