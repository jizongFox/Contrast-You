from typing import Union

import torch
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.meters2 import MeterInterface, AverageValueMeter
from torch import nn

from contrastyou.losses.contrast_loss2 import SelfPacedSupConLoss, SupConLoss1
from semi_seg.arch.hook import SingleFeatureExtractor
from semi_seg.hooks.base import TrainHook, EpochHook
from semi_seg.mi_estimator.base import decoder_names, encoder_names


class INFONCEHook(TrainHook):

    def __init__(self, *, name, model: nn.Module, feature_name: str, weight: float = 1.0, spatial_size=(1, 1),
                 contrastive_criterion: Union[SupConLoss1, SelfPacedSupConLoss]) -> None:
        super().__init__(hook_name=name)
        assert feature_name in encoder_names + decoder_names, feature_name
        self._feature_name = feature_name
        self._weight = weight

        self._extractor = SingleFeatureExtractor(model, feature_name=feature_name)  # noqa
        input_dim = model.get_channel_dim(feature_name)
        self._projector = self.init_projector(input_dim=input_dim, spatial_size=spatial_size)
        self._criterion = self.init_criterion()

        self._contrastive_criterion = contrastive_criterion

        self._learnable_models = (self._projector,)

    def __call__(self):
        return _INFONCEEpochHook(name=self._hook_name, weight=self._weight, extractor=self._extractor,
                                 projector=self._projector,
                                 criterion=self._criterion)

    def init_projector(self, *, input_dim, spatial_size=(1, 1)):
        projector = self.projector_class(input_dim=input_dim, hidden_dim=256, output_dim=256, head_type="mlp",
                                         normalize=True, spatial_size=spatial_size)
        return projector

    def init_criterion(self):
        if self._feature_name in encoder_names:
            return self._init_criterion()
        return self._init_dense_criterion()

    def _init_dense_criterion(self):
        raise NotImplementedError()

    def _init_criterion(self):
        criterion = self.criterion_class()
        return criterion

    @property
    def projector_class(self):
        from contrastyou.projectors.heads import ProjectionHead, DenseProjectionHead
        if self._feature_name in encoder_names:
            return ProjectionHead
        return DenseProjectionHead

    @property
    def criterion_class(self):
        from contrastyou.losses.iic_loss import IIDLoss, IIDSegmentationLoss
        if self._feature_name in encoder_names:
            return IIDLoss
        return IIDSegmentationLoss


class _INFONCEEpochHook(EpochHook):

    def __init__(self, *, name: str, weight: float, extractor, projector, criterion) -> None:
        super().__init__(name)
        self._extractor = extractor
        self._extractor.bind()
        self._extractor.set_enable(True)
        self._weight = weight
        self._projector = projector
        self._criterion = criterion
        self.meters = MeterInterface()
        self.configure_meters(self.meters)

    @staticmethod
    def configure_meters(meters):
        meters.register_meter("mi", AverageValueMeter())

    def before_forward_pass(self, **kwargs):
        self._extractor.clear()

    def __call__(self, *, affine_transformer, seed, **kwargs):
        feature_ = self._extractor.feature()
        proj_feature, proj_tf_feature = torch.chunk(feature_, 2, dim=0)
        with FixRandomSeed(seed):
            proj_feature_tf = affine_transformer(proj_feature)

        self.meters["mi"].add(loss.item())

        return self._criterion(proj_feature_tf, proj_tf_feature, target=labels)
