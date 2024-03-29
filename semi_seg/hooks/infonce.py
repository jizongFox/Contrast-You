import typing as t
from functools import partial, lru_cache
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from loguru import logger
from torch import nn, Tensor
from torch.nn import functional as F

from contrastyou.arch.utils import SingleFeatureExtractor
from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.losses.contrastive import SelfPacedSupConLoss, SupConLoss1
from contrastyou.meters import MeterInterface, AverageValueMeter
from contrastyou.utils import fix_all_seed_for_transforms, switch_plt_backend
from contrastyou.writer import get_tb_writer
from .utils import get_label


@lru_cache()
def _load_superpixel_image(f: str) -> Tensor:
    f = list(Path(f).parts)
    f.insert(-1, "superpixel")
    with Image.open(Path(*f).as_posix() + ".png") as img:
        return torch.from_numpy(np.array(img.convert("L"), dtype=np.float32))  # noqa


def region_extractor(normalize_features, *, point_nums=5, seed: int):
    """
    extractor for dense features, used for contrastive training.
    """

    def get_feature_selected(feature_map, n_point_coordinate):
        return torch.stack([feature_map[:, n[0], n[1]] for n in n_point_coordinate], dim=0)

    def get_n_point_coordinate(h, w, n):
        return [(x, y) for x, y in zip(np.random.choice(range(h), n, replace=False),
                                       np.random.choice(range(w), n, replace=False))]

    with fix_all_seed_for_transforms(seed):
        h, w = normalize_features.shape[2:]
        return torch.cat([get_feature_selected(single_feature, get_n_point_coordinate(n=point_nums, h=h, w=w))
                          for single_feature in normalize_features], dim=0)


@switch_plt_backend("agg")
def figure2board(tensor, name, criterion, writer, epocher):
    fig1 = plt.figure()
    plt.imshow(tensor.detach().float().cpu().numpy(), cmap="gray")
    plt.colorbar()
    dest = "/".join([criterion.__class__.__name__, name])
    writer.add_figure(tag=dest, figure=fig1, global_step=epocher._cur_epoch)  # noqa


class PScheduler:
    """
    scheduler function for the gamma as the self-paced loss.
    """

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
    """
    Contrastive hook for each layer.
    """

    @property
    def learnable_modules(self) -> List[nn.Module]:
        return [self._projector, ]

    def __init__(self, *, name, model: nn.Module, feature_name: str, weight: float = 1.0,
                 spatial_size: t.Sequence[int] = None,
                 data_name: str, contrast_on: str) -> None:
        super().__init__(hook_name=name)
        self.register_non_trackable_buffer("_model", model)
        assert feature_name in model.arch_elements
        self._feature_name = feature_name
        self._weight = weight

        self._extractor = SingleFeatureExtractor(model, feature_name=feature_name)  # noqa
        input_dim = model.get_channel_dim(feature_name)
        if feature_name in self._model.encoder_names:
            assert (spatial_size is None) or (spatial_size == (1, 1))
            spatial_size = (1, 1)
        else:
            assert isinstance(spatial_size, t.Sequence) and isinstance(tuple(spatial_size)[0], int)
        self._projector = self.init_projector(input_dim=input_dim, spatial_size=spatial_size)
        self._criterion = self.init_criterion()
        self._label_generator = partial(get_label, contrast_on=contrast_on, data_name=data_name)

    def __call__(self):
        if self.is_encoder:
            hook = _INFONCEEpochHook(
                name=self._hook_name, weight=self._weight, extractor=self._extractor, projector=self._projector,
                criterion=self._criterion, label_generator=self._label_generator
            )
            return hook
        return _INFONCEDenseHook(
            name=self._hook_name, weight=self._weight, extractor=self._extractor, projector=self._projector,
            criterion=self._criterion, label_generator=self._label_generator
        )

    def init_criterion(self) -> SupConLoss1:
        self._criterion = SupConLoss1()
        return self._criterion

    def init_projector(self, *, input_dim, spatial_size):
        projector = self.projector_class(input_dim=input_dim, hidden_dim=256, output_dim=256, head_type="mlp",
                                         normalize=True, spatial_size=spatial_size)
        return projector

    @property
    def projector_class(self):
        from contrastyou.projectors.heads import ProjectionHead, DenseProjectionHead
        if self.is_encoder:
            return ProjectionHead
        return DenseProjectionHead

    @property
    def is_encoder(self):
        return self._feature_name in self._model.encoder_names


class SelfPacedINFONCEHook(INFONCEHook):
    """
    self-paced contrastive loss for each layer
    """

    def __init__(self, *, name, model: nn.Module, feature_name: str, weight: float = 1.0, spatial_size=(1, 1),
                 data_name: str, contrast_on: str, mode="soft", p=0.5, begin_value=1e6, end_value=1e6,
                 correct_grad: bool = False, max_epoch: int) -> None:
        self._mode = mode
        self._p = float(p)
        self._begin_value = float(begin_value)
        self._end_value = float(end_value)
        self._max_epoch = int(max_epoch)
        self._correct_grad = correct_grad
        super().__init__(name=name, model=model, feature_name=feature_name, weight=weight, spatial_size=spatial_size,
                         data_name=data_name, contrast_on=contrast_on)

    def init_criterion(self) -> SelfPacedSupConLoss:
        self._scheduler = PScheduler(max_epoch=self._max_epoch, begin_value=self._begin_value,
                                     end_value=self._end_value, p=self._p)
        self._criterion = SelfPacedSupConLoss(weight_update=self._mode, correct_grad=self._correct_grad)
        return self._criterion

    def __call__(self):
        gamma = self._scheduler.value
        self._scheduler.step()
        self._criterion.set_gamma(gamma)
        hook = _SPINFONCEEpochHook(
            name=self._hook_name, weight=self._weight, extractor=self._extractor, projector=self._projector,
            criterion=self._criterion, label_generator=self._label_generator
        )
        return hook


class SuperPixelInfoNCEHook(INFONCEHook):

    def __init__(self, *, name, model: nn.Module, feature_name: str, weight: float = 1.0,
                 spatial_size: t.Sequence[int] = None, data_name: str, contrast_on: str) -> None:
        logger.debug("SuperPixelInfoNCEHook initializing")
        super().__init__(name=name, model=model, feature_name=feature_name, weight=weight, spatial_size=spatial_size,
                         data_name=data_name, contrast_on=contrast_on)
        assert self.is_encoder is False, f"{self.__class__.__name__} only supports decoder features"

    def __call__(self) -> "_SuperPixelInfoNCEEPochHook":
        return _SuperPixelInfoNCEEPochHook(
            name=self._hook_name, weight=self._weight, extractor=self._extractor, projector=self._projector,
            criterion=self._criterion, label_generator=self._label_generator
        )


class _INFONCEEpochHook(EpocherHook):

    def __init__(self, *, name: str, weight: float, extractor, projector,
                 criterion: Union[SupConLoss1, SelfPacedSupConLoss],
                 label_generator) -> None:
        super().__init__(name=name)
        self._extractor = extractor
        self._extractor.bind()
        self._weight = weight
        self._projector = projector
        self._criterion = criterion
        self._label_generator = label_generator
        self._n = 0

    def configure_meters_given_epocher(self, meters: MeterInterface):
        meters = super().configure_meters_given_epocher(meters)
        meters.register_meter("loss", AverageValueMeter())
        return meters

    def before_forward_pass(self, **kwargs):
        self._extractor.clear()
        self._extractor.set_enable(True)

    def after_forward_pass(self, **kwargs):
        self._extractor.set_enable(False)

    def _call_implementation(self, *, affine_transformer, seed, unlabeled_tf_logits, unlabeled_logits_tf,
                             partition_group,
                             label_group, **kwargs):
        n_unl = len(unlabeled_logits_tf)
        feature_ = self._extractor.feature()[-n_unl * 2:]
        unlabeled_features, unlabeled_tf_features = torch.chunk(feature_, 2, dim=0)
        unlabeled_features_tf = affine_transformer(unlabeled_features)
        norm_features_tf, norm_tf_features = torch.chunk(
            self._projector(torch.cat([unlabeled_features_tf, unlabeled_tf_features], dim=0)), 2)
        labels = self._label_generator(partition_group=partition_group, label_group=label_group)
        loss = self._criterion(norm_features_tf, norm_tf_features, target=labels)
        self.meters["loss"].add(loss.item())

        if self._n == 0:
            sim_exp = self._criterion.sim_exp
            sim_logits = self._criterion.sim_logits
            pos_mask = self._criterion.pos_mask
            writer = get_tb_writer()
            figure2board(pos_mask, self.name + "/mask", self._criterion, writer, self.epocher)
            figure2board(sim_exp, self.name + "/sim_exp", self._criterion, writer, self.epocher)
            figure2board(sim_logits, self.name + "/sim_logits", self._criterion, writer, self.epocher)

        self._n += 1
        return loss * self._weight

    def close(self):
        self._extractor.remove()


class _INFONCEDenseHook(_INFONCEEpochHook):

    def _call_implementation(self, *, affine_transformer, seed, unlabeled_tf_logits, unlabeled_logits_tf,
                             partition_group,
                             label_group, **kwargs):
        n_unl = len(unlabeled_logits_tf)
        feature_ = self._extractor.feature()[-n_unl * 2:]
        unlabeled_features, unlabeled_tf_features = torch.chunk(feature_, 2, dim=0)
        unlabeled_features_tf = affine_transformer(unlabeled_features, seed=seed)
        norm_features_tf, norm_tf_features = torch.chunk(
            self._projector(torch.cat([unlabeled_features_tf, unlabeled_tf_features], dim=0)), 2)
        norm_features_tf_selected = region_extractor(norm_features_tf, point_nums=5, seed=seed)
        norm_tf_features_selected = region_extractor(norm_tf_features, point_nums=5, seed=seed)

        labels = list(range(norm_features_tf_selected.shape[0]))
        loss = self._criterion(norm_features_tf_selected, norm_tf_features_selected, target=labels)
        self.meters["loss"].add(loss.item())

        if self._n == 0:
            sim_exp = self._criterion.sim_exp
            sim_logits = self._criterion.sim_logits
            pos_mask = self._criterion.pos_mask
            writer = get_tb_writer()
            figure2board(pos_mask, self.name + "/mask", self._criterion, writer, self.epocher)
            figure2board(sim_exp, self.name + "/sim_exp", self._criterion, writer, self.epocher)
            figure2board(sim_logits, self.name + "/sim_logits", self._criterion, writer, self.epocher)

        self._n += 1
        return loss * self._weight


class _SPINFONCEEpochHook(_INFONCEEpochHook):
    _criterion: SelfPacedSupConLoss

    def configure_meters_given_epocher(self, meters: MeterInterface):
        meters = super().configure_meters_given_epocher(meters)
        meters.register_meter("sp_weight", AverageValueMeter())
        meters.register_meter("age_param", AverageValueMeter())
        return meters

    def _call_implementation(self, *, affine_transformer, seed, unlabeled_tf_logits, unlabeled_logits_tf,
                             partition_group,
                             label_group, **kwargs):
        loss = super()._call_implementation(
            affine_transformer=affine_transformer, seed=seed,
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf,
            partition_group=partition_group, label_group=label_group, **kwargs)
        self.meters["sp_weight"].add(self._criterion.downgrade_ratio)
        self.meters["age_param"].add(self._criterion.age_param)

        if self._n == 1:
            sp_mask = self._criterion.sp_mask.float()
            writer = get_tb_writer()
            figure2board(sp_mask, "sp_mask", self._criterion, writer, self.epocher)

        return loss


class _SuperPixelInfoNCEEPochHook(_INFONCEEpochHook):
    def _call_implementation(self, *,
                             affine_transformer,
                             seed,
                             unlabeled_tf_logits,
                             unlabeled_logits_tf,
                             partition_group,
                             label_group,
                             batch_data: t.Dict[str, t.Any] = None,
                             **kwargs):
        assert batch_data is not None
        n_unl = len(unlabeled_logits_tf)
        feature_ = self._extractor.feature()[-n_unl * 2:]
        unlabeled_features, unlabeled_tf_features = torch.chunk(feature_, 2, dim=0)
        unlabeled_features_tf = affine_transformer(unlabeled_features, seed=seed)
        norm_features_tf, norm_tf_features = torch.chunk(
            self._projector(torch.cat([unlabeled_features_tf, unlabeled_tf_features], dim=0)), 2)

        norm_features_tf_selected = region_extractor(norm_features_tf, point_nums=5, seed=seed)
        norm_tf_features_selected = region_extractor(norm_tf_features, point_nums=5, seed=seed)

        superpixel_mask = (batch_data["superpixel"][0].to(norm_tf_features.device) * 255.0).type(torch.uint8).float()

        superpixel_mask_tf = affine_transformer(superpixel_mask)

        superpixel_mask_tf_pooled = F.interpolate(
            superpixel_mask_tf,
            size=(norm_features_tf.shape[-2], norm_features_tf.shape[-1]),
            mode='nearest'
        )
        superpixel_mask_selected = region_extractor(superpixel_mask_tf_pooled, point_nums=5, seed=seed)

        labels = superpixel_mask_selected.squeeze().type(torch.uint8).tolist()
        loss = self._criterion(norm_features_tf_selected, norm_tf_features_selected, target=labels)
        self.meters["loss"].add(loss.item())

        if self._n == 0:
            sim_exp = self._criterion.sim_exp
            sim_logits = self._criterion.sim_logits
            pos_mask = self._criterion.pos_mask
            writer = get_tb_writer()
            figure2board(pos_mask, self.name + "/mask", self._criterion, writer, self.epocher)
            figure2board(sim_exp, self.name + "/sim_exp", self._criterion, writer, self.epocher)
            figure2board(sim_logits, self.name + "/sim_logits", self._criterion, writer, self.epocher)

        self._n += 1
        return loss * self._weight
