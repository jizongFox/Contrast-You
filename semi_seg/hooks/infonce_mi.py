from functools import partial
from typing import List

import numpy as np
import torch
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.schedulers.customized_scheduler import WeightScheduler
from torch import nn

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.losses.contrast_loss3 import SelfPacedSupConLoss, SupConLoss1
from contrastyou.meters import MeterInterface, AverageValueMeter
from semi_seg.arch.hook import SingleFeatureExtractor
from semi_seg.mi_estimator.base import decoder_names, encoder_names
from .utils import get_label


class PScheduler(WeightScheduler):
    def __init__(self, max_epoch, begin_value=0.0, end_value=1.0, p=0.5):
        super().__init__()
        self.max_epoch = max_epoch
        self.begin_value = float(begin_value)
        self.end_value = float(end_value)
        self.epoch = 0
        self.p = p

    def step(self):
        self.epoch += 1

    @property
    def value(self):
        return self.get_lr(self.epoch)

    def get_lr(self, cur_epoch):
        return self.begin_value + (self.end_value - self.begin_value) * np.power(
            cur_epoch / self.max_epoch, self.p
        )


class INFONCEHook(TrainerHook):

    @property
    def learnable_modules(self) -> List[nn.Module]:
        return [self._projector, ]

    def __init__(self, *, name, model: nn.Module, feature_name: str, weight: float = 1.0, spatial_size=(1, 1),
                 data_name: str, contrast_on: str) -> None:
        super().__init__(hook_name=name)
        assert feature_name in encoder_names + decoder_names, feature_name
        self._feature_name = feature_name
        self._weight = weight

        self._extractor = SingleFeatureExtractor(model, feature_name=feature_name)  # noqa
        input_dim = model.get_channel_dim(feature_name)
        self._projector = self.init_projector(input_dim=input_dim, spatial_size=spatial_size)
        self._criterion = self.init_criterion()
        self._label_generator = partial(get_label, contrast_on=contrast_on, data_name=data_name)
        self._learnable_models = (self._projector,)

    def __call__(self):
        hook = _INFONCEEpochHook(
            name=self._hook_name, weight=self._weight, extractor=self._extractor, projector=self._projector,
            criterion=self._criterion, label_generator=self._label_generator
        )
        return hook

    def init_criterion(self):
        self._criterion = SupConLoss1()
        return self._criterion

    def init_projector(self, *, input_dim, spatial_size=(1, 1)):
        projector = self.projector_class(input_dim=input_dim, hidden_dim=256, output_dim=256, head_type="mlp",
                                         normalize=True, spatial_size=spatial_size)
        return projector

    @property
    def projector_class(self):
        from contrastyou.projectors.heads import ProjectionHead, DenseProjectionHead
        if self._feature_name in encoder_names:
            return ProjectionHead
        return DenseProjectionHead


class SelfPacedINFONCEHook(INFONCEHook):

    def __init__(self, *, name, model: nn.Module, feature_name: str, weight: float = 1.0, spatial_size=(1, 1),
                 data_name: str, contrast_on: str, mode="soft", p=0.5, begin_value=1e6, end_value=1e6,
                 max_epoch: int) -> None:
        self._mode = mode
        self._p = float(p)
        self._begin_value = float(begin_value)
        self._end_value = float(end_value)
        self._max_epoch = int(max_epoch)
        super().__init__(name=name, model=model, feature_name=feature_name, weight=weight, spatial_size=spatial_size,
                         data_name=data_name, contrast_on=contrast_on)

    def init_criterion(self):
        self._scheduler = PScheduler(max_epoch=self._max_epoch, begin_value=self._begin_value,
                                     end_value=self._end_value, p=self._p)
        self._criterion = SelfPacedSupConLoss(weight_update=self._mode)
        return self._criterion

    def __call__(self):
        gamma = self._scheduler.value
        self._scheduler.step()
        self._criterion.set_gamma(gamma)
        return super().__call__()


class _INFONCEEpochHook(EpocherHook):

    def __init__(self, *, name: str, weight: float, extractor, projector, criterion, label_generator) -> None:
        super().__init__(name)
        self._extractor = extractor
        self._extractor.bind()
        self._weight = weight
        self._projector = projector
        self._criterion = criterion
        self._label_generator = label_generator

    def configure_meters(self, meters: MeterInterface):
        meters = super().configure_meters(meters)
        with meters.focus_on(self._name):
            meters.register_meter("loss", AverageValueMeter())

    def before_forward_pass(self, **kwargs):
        self._extractor.clear()
        self._extractor.set_enable(True)

    def after_forward_pass(self, **kwargs):
        self._extractor.set_enable(False)

    def __call__(self, *, affine_transformer, seed, unlabeled_tf_logits, unlabeled_logits_tf, partition_group,
                 label_group, **kwargs):
        n_unl = len(unlabeled_logits_tf)
        feature_ = self._extractor.feature()[-n_unl * 2:]
        proj_feature, proj_tf_feature = torch.chunk(feature_, 2, dim=0)
        with FixRandomSeed(seed):
            proj_feature_tf = torch.stack([affine_transformer(x) for x in proj_feature], dim=0)
        norm_feature_tf, norm_tf_feature = torch.chunk(
            self._projector(torch.cat([proj_feature_tf, proj_tf_feature], dim=0)), 2)
        labels = self._label_generator(partition_group=partition_group, label_group=label_group)
        loss = self._criterion(norm_feature_tf, norm_tf_feature, target=labels)
        with self.meters.focus_on(self._name):
            self.meters["loss"].add(loss.item())
        return loss * self._weight
