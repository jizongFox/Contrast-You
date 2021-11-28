import random
import typing as t
import warnings
import weakref
from itertools import chain

import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn import functional as F

from contrastyou.arch.unet import UNetFeatureMapEnum
from contrastyou.arch.utils import SingleFeatureExtractor
from contrastyou.hooks import TrainerHook, EpocherHook
from contrastyou.losses.cc import CCLoss
from contrastyou.losses.discreteMI import IIDSegmentationLoss
from contrastyou.losses.kl import Entropy
from contrastyou.meters import AverageValueMeter
from contrastyou.projectors import CrossCorrelationProjector
from contrastyou.utils import class_name, average_iter, item2str, probs2one_hot
from semi_seg.hooks.utils import FeatureMapSaver

if t.TYPE_CHECKING:
    from contrastyou.projectors.nn import _ProjectorHeadBase  # noqa
    from contrastyou.meters import MeterInterface

__all__ = ["CrossCorrelationHook", "CrossCorrelationHookWithSaver", "ProjectorGeneralHook", "_CrossCorrelationHook",
           "_MIHook"]


class CrossCorrelationHook(TrainerHook):

    def __init__(self, *, name: str, model: nn.Module, feature_name: UNetFeatureMapEnum, cc_weight: float,
                 mi_weight: float = 0.0, kernel_size: int, projector_params: t.Dict[str, t.Any],
                 adding_coordinates: bool, mi_criterion_params: t.Dict[str, t.Any],
                 norm_params: t.Dict[str, t.Any]):
        super().__init__(hook_name=name)
        self._cc_weight = float(cc_weight)
        self._mi_weight = float(mi_weight)
        feature_name = UNetFeatureMapEnum(feature_name)
        self._feature_name = feature_name.value
        logger.info(
            f"Creating {class_name(self)} @{feature_name.name}.")
        self._extractor = SingleFeatureExtractor(
            model=model, feature_name=UNetFeatureMapEnum(feature_name).name  # noqa
        )
        self.adding_coordinates = adding_coordinates
        input_dim = model.get_channel_dim(feature_name.value)  # model: type: UNet
        if self.adding_coordinates:
            input_dim += 2
        logger.trace(f"Creating projector with {item2str(projector_params)}")
        self._projector = CrossCorrelationProjector(input_dim=input_dim, **projector_params)

        logger.trace(f"Creating CCLoss with kernel_size = {kernel_size} with weight = {self._cc_weight}.")
        self._cc_criterion = CCLoss(win=(kernel_size, kernel_size))

        logger.trace(f"Creating IIDSegmentationLoss with kernel_size = {kernel_size} with weight = {self._mi_weight}.")
        self._mi_criterion = IIDSegmentationLoss(**mi_criterion_params)

        self._diff_power: float = float(norm_params["power"])
        assert 0 <= self._diff_power <= 1, self._diff_power
        self._use_image_diff: bool = norm_params["image_diff"]

    def __call__(self, **kwargs):
        return _CrossCorrelationEpocherHook(
            name=self._hook_name, extractor=self._extractor,
            projector=self._projector, cc_criterion=self._cc_criterion,
            cc_weight=self._cc_weight, mi_weight=self._mi_weight,
            mi_criterion=self._mi_criterion, diff_power=self.diff_power, add_coordinates=self.adding_coordinates,
            image_diff=self._use_image_diff
        )

    @property
    def learnable_modules(self) -> t.List[nn.Module]:
        return [self._projector, ]


class _CrossCorrelationEpocherHook(EpocherHook):

    def __init__(self, *, name: str = "cc", extractor: 'SingleFeatureExtractor', projector: '_ProjectorHeadBase',
                 cc_criterion: CCLoss, mi_criterion: 'IIDSegmentationLoss', add_coordinates: bool,
                 cc_weight: float, mi_weight: float, diff_power: float = 1.0, image_diff: bool) -> None:
        super().__init__(name=name)
        self.cc_weight = cc_weight
        self.mi_weight = mi_weight
        self.extractor = extractor
        self.extractor.bind()
        self.projector = projector
        self.cc_criterion = cc_criterion
        self.mi_criterion = mi_criterion
        self._ent_func = Entropy(reduction="none")
        self._diff_power = diff_power
        self._image_diff = image_diff  # this is to check whether use edge detection.
        self.add_coordinates = add_coordinates

    def close(self):
        self.extractor.remove()

    def configure_meters_given_epocher(self, meters: 'MeterInterface'):
        meters.register_meter("cc_ls", AverageValueMeter())
        meters.register_meter("mi_ls", AverageValueMeter())
        return meters

    def before_forward_pass(self, **kwargs):
        self.extractor.clear()
        self.extractor.set_enable(True)

    def after_forward_pass(self, **kwargs):
        self.extractor.set_enable(False)

    def _call_implementation(
            self, unlabeled_image_tf: Tensor, unlabeled_logits_tf: Tensor,
            affine_transformer: t.Callable[[Tensor], Tensor],
            unlabeled_image: Tensor, **kwargs
    ):
        n_unl = len(unlabeled_logits_tf)
        feature_ = self.extractor.feature()[-n_unl * 2:]
        _unlabeled_features, unlabeled_tf_features = torch.chunk(feature_, 2, dim=0)
        unlabeled_features_tf = affine_transformer(_unlabeled_features)

        if self.add_coordinates:
            unlabeled_tf_features, unlabeled_features_tf = self.merge_coordinate(
                unlabeled_tf_features=unlabeled_tf_features,
                unlabeled_features_tf=unlabeled_features_tf
            )

        projected_dist_tf, projected_tf_dist = zip(*[torch.chunk(x, 2) for x in self.projector(
            torch.cat([unlabeled_features_tf, unlabeled_tf_features], dim=0))])

        losses, diff_image, diff_prediction = zip(*[
            self.cc_loss_per_head(image=unlabeled_image_tf, predict_simplex=x) for x in
            chain(projected_dist_tf, projected_tf_dist)
        ])
        cc_loss = average_iter(losses)
        mi_loss = self.mi_loss_per_head(projected_dist_tf, projected_tf_dist)
        if self.meters:
            self.meters["cc_ls"].add(cc_loss.item())
            self.meters["mi_ls"].add(mi_loss.item())
        return cc_loss * self.cc_weight + mi_loss * self.mi_weight

    def norm(self, image: Tensor, min=0.0, max=1.0, slicewise=True):
        if not slicewise:
            return self._norm(image, min, max)
        return torch.stack([self._norm(x) for x in image], dim=0)

    def _norm(self, image: Tensor, min=0.0, max=1.0):
        min_, max_ = image.min().detach(), image.max().detach()
        image = image - min_
        image = image / (max_ - min_ + 1e-6)
        return image * (max - min) + min

    @staticmethod
    def diff(image: Tensor):
        assert image.dim() == 4
        dx = image - torch.roll(image, shifts=1, dims=2)
        dy = image - torch.roll(image, shifts=1, dims=3)
        d = torch.sqrt(dx.pow(2) + dy.pow(2))
        return torch.mean(d, dim=1, keepdims=True)  # noqa

    def cc_loss_per_head(self, image: Tensor, predict_simplex: Tensor):
        if tuple(image.shape[-2:]) != tuple(predict_simplex.shape[-2:]):
            h, w = predict_simplex.shape[-2:]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                image = F.interpolate(image, size=(h, w), mode="bilinear")
        # check if image_diff works in some sense.
        if self._image_diff:
            diff_image = self.norm(self.diff(image), min=0, max=1).pow(
                self._diff_power)  # the diff power applies only on edges.
        else:
            diff_image = image

        diff_tf_softmax = self.norm(self._ent_func(predict_simplex), min=0, max=1, slicewise=False).unsqueeze(1)

        loss = self.cc_criterion(
            diff_tf_softmax,
            diff_image
        )
        return loss, diff_image, diff_tf_softmax

    def mi_loss_per_head(self, prob1, prob2):
        loss = sum([self.mi_criterion(x1, x2) for x1, x2 in zip(prob1, prob2)]) / len(prob1)
        return loss

    @staticmethod
    def merge_coordinate(*, unlabeled_tf_features: Tensor, unlabeled_features_tf: Tensor):
        (width, height), bn = unlabeled_features_tf.shape[-2:], unlabeled_features_tf.shape[0]
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(start=-1, end=1, steps=width, device=unlabeled_features_tf.device,
                           dtype=unlabeled_features_tf.dtype),
            torch.linspace(start=-1, end=1, steps=height, device=unlabeled_features_tf.device,
                           dtype=unlabeled_features_tf.dtype), indexing='ij')
        coordinate = torch.stack([grid_x, grid_y], dim=0)[None, ...].repeat(bn, 1, 1, 1)
        unlabeled_features_tf = torch.cat([unlabeled_features_tf, coordinate], dim=1)
        unlabeled_tf_features = torch.cat([unlabeled_tf_features, coordinate], dim=1)
        return unlabeled_tf_features, unlabeled_features_tf


class CrossCorrelationHookWithSaver(CrossCorrelationHook):
    # with an image saver

    def __init__(self, *, name: str, model: nn.Module, feature_name: UNetFeatureMapEnum, cc_weight: float,
                 mi_weight: float = 0.0, kernel_size: int, projector_params: t.Dict[str, t.Any],
                 adding_coordinates: bool, mi_criterion_params: t.Dict[str, t.Any], norm_params: t.Dict[str, t.Any],
                 save: bool = False):
        super().__init__(
            name=name, model=model, feature_name=feature_name, cc_weight=cc_weight, mi_weight=mi_weight,
            kernel_size=kernel_size, projector_params=projector_params, adding_coordinates=adding_coordinates,
            mi_criterion_params=mi_criterion_params,
            norm_params=norm_params,
        )
        self.save = save
        self.saver = None

    def after_initialize(self):
        if self.save:
            self.saver = FeatureMapSaver(save_dir=self.trainer.absolute_save_dir, folder_name=f"vis/{self._hook_name}")

    def __call__(self, **kwargs):
        if not self.save:
            return super(CrossCorrelationHookWithSaver, self).__call__(**kwargs)
        return _CrossCorrelationEpocherHookWithSaver(
            name=self._hook_name, extractor=self._extractor,
            projector=self._projector, cc_criterion=self._cc_criterion,
            cc_weight=self._cc_weight, mi_weight=self._mi_weight,
            mi_criterion=self._mi_criterion, saver=self.saver,
            diff_power=self._diff_power, adding_coordinates=self.adding_coordinates,
            image_diff=self._use_image_diff
        )

    def close(self):
        if self.saver:
            self.saver.zip()


class _CrossCorrelationEpocherHookWithSaver(_CrossCorrelationEpocherHook):
    # with an image saver

    def __init__(self, *, name: str = "cc", extractor: 'SingleFeatureExtractor', projector: '_ProjectorHeadBase',
                 cc_criterion: 'CCLoss', mi_criterion: 'IIDSegmentationLoss', adding_coordinates: bool,
                 cc_weight: float, mi_weight: float, diff_power: float, image_diff: bool,
                 saver: 'FeatureMapSaver') -> None:
        super().__init__(name=name, extractor=extractor, projector=projector, cc_criterion=cc_criterion,
                         add_coordinates=adding_coordinates,
                         mi_criterion=mi_criterion, cc_weight=cc_weight, mi_weight=mi_weight, diff_power=diff_power,
                         image_diff=image_diff)
        self.saver = saver

    def _call_implementation(
            self, unlabeled_image_tf: Tensor, unlabeled_logits_tf: Tensor,
            affine_transformer: t.Callable[[Tensor], Tensor],
            unlabeled_image: Tensor, **kwargs
    ):
        n_unl = len(unlabeled_logits_tf)
        feature_ = self.extractor.feature()[-n_unl * 2:]
        _unlabeled_features, unlabeled_tf_features = torch.chunk(feature_, 2, dim=0)
        unlabeled_features_tf = affine_transformer(_unlabeled_features)
        save_image_condition = self.epocher.cur_batch_num == 0 and self.epocher.cur_epoch % 5 == 0
        if save_image_condition:
            self.saver.save_map(
                image=unlabeled_image_tf, feature_map1=unlabeled_tf_features, feature_map2=unlabeled_features_tf,
                cur_epoch=self.epocher.cur_epoch, cur_batch_num=self.epocher.cur_batch_num, save_name="feature"
            )

        if self.add_coordinates:
            unlabeled_tf_features, unlabeled_features_tf = self.merge_coordinate(
                unlabeled_tf_features=unlabeled_tf_features,
                unlabeled_features_tf=unlabeled_features_tf
            )
        projected_dist_tf, projected_tf_dist = zip(
            *[
                torch.chunk(x, 2) for x in
                self.projector(torch.cat([unlabeled_features_tf, unlabeled_tf_features], dim=0))
            ]
        )
        if save_image_condition:
            self.saver.save_map(
                image=unlabeled_image_tf, feature_map1=projected_dist_tf[0], feature_map2=projected_tf_dist[0],
                cur_epoch=self.epocher.cur_epoch, cur_batch_num=self.epocher.cur_batch_num, save_name="probability"
            )

        losses, diff_image, diff_prediction = zip(*[
            self.cc_loss_per_head(image=unlabeled_image_tf, predict_simplex=x) for x in
            chain(projected_dist_tf, projected_tf_dist)
        ])
        if save_image_condition:
            self.saver.save_map(
                image=diff_image[0], feature_map1=diff_prediction[0], feature_map2=diff_prediction[0],
                cur_epoch=self.epocher.cur_epoch, cur_batch_num=self.epocher.cur_batch_num,
                save_name="cross_correlation", feature_type="image"
            )

        cc_loss = average_iter(losses)
        mi_loss = self.mi_loss_per_head(projected_dist_tf, projected_tf_dist)
        if self.meters:
            self.meters["cc_ls"].add(cc_loss.item())
            self.meters["mi_ls"].add(mi_loss.item())
        return cc_loss * self.cc_weight + mi_loss * self.mi_weight


# new interface
class _TinyHook:

    def __init__(self, *, name: str, criterion: t.Callable, weight: float) -> None:
        self.name = name
        self.criterion = criterion
        self.weight = weight
        self.meters = None
        self.hook = None
        logger.trace(f"Created {self.__class__.__name__}:{self.name} with weight={self.weight}.")

    def configure_meters(self, meters: 'MeterInterface'):
        meters.register_meter(self.name, AverageValueMeter())
        return meters

    def __call__(self, **kwargs) -> Tensor:
        loss = self.criterion(**kwargs)
        if self.meters:
            self.meters[self.name].add(loss.item())
        return loss * self.weight

    def close(self):
        pass


class ProjectorGeneralHook(TrainerHook):

    def __init__(self, *, name: str, model: nn.Module, feature_name: UNetFeatureMapEnum,
                 projector_params: t.Dict[str, t.Any], save: bool = False):
        super().__init__(hook_name=name)
        feature_name = UNetFeatureMapEnum(feature_name)
        self._feature_name = feature_name.value
        logger.info(
            f"Creating {class_name(self)} @{feature_name.name}.")
        self._extractor = SingleFeatureExtractor(
            model=model, feature_name=UNetFeatureMapEnum(feature_name).name  # noqa
        )
        input_dim = model.get_channel_dim(feature_name.value)  # model: type: UNet
        logger.trace(f"Creating projector with {item2str(projector_params)}")
        self._projector = CrossCorrelationProjector(input_dim=input_dim, **projector_params)

        self._feature_hooks = []
        self._dist_hooks = []
        self.save = save
        self.saver = None

    def after_initialize(self):
        if self.save:
            self.saver = FeatureMapSaver(save_dir=self.trainer.absolute_save_dir,
                                         folder_name=f"vis/{self._hook_name}")

    def register_feat_hook(self, *hook: '_TinyHook'):
        logger.debug(f"register {hook}")
        self._feature_hooks.extend(hook)

    def register_dist_hook(self, *hook: '_TinyHook'):
        logger.debug(f"register {hook}")
        self._dist_hooks.extend(hook)

    def __call__(self, **kwargs):
        if (len(self._feature_hooks) + len(self._dist_hooks)) == 0:
            raise RuntimeError(f"hooks not registered for {class_name(self)}.")

        return _ProjectorEpocherGeneralHook(
            name=self._hook_name, extractor=self._extractor, projector=self._projector, dist_hooks=self._dist_hooks,
            feat_hooks=self._feature_hooks, saver=self.saver
        )

    @property
    def learnable_modules(self) -> t.List[nn.Module]:
        return [self._projector, ]

    def close(self):
        if self.save:
            self.saver.zip()


# try to convert this to hook
class _ProjectorEpocherGeneralHook(EpocherHook):

    def __init__(self, *, name: str, extractor: 'SingleFeatureExtractor', projector: '_ProjectorHeadBase',
                 dist_hooks: t.Sequence[_TinyHook] = (), feat_hooks: t.Sequence[_TinyHook] = (), saver=None, ) -> None:
        super().__init__(name=name)
        self.extractor = extractor
        self.extractor.bind()
        self.projector = projector
        self._feature_hooks = feat_hooks
        self._dist_hooks = dist_hooks
        self.saver = saver

    def configure_meters_given_epocher(self, meters: 'MeterInterface'):
        meters = super(_ProjectorEpocherGeneralHook, self).configure_meters_given_epocher(meters)
        for h in chain(self._dist_hooks, self._feature_hooks):
            h.meters = meters
            h.hook = weakref.proxy(self)
            h.configure_meters(meters)
        return meters

    def before_forward_pass(self, **kwargs):
        self.extractor.clear()
        self.extractor.set_enable(True)

    def after_forward_pass(self, **kwargs):
        self.extractor.set_enable(False)

    def _call_implementation(self, unlabeled_image_tf: Tensor, unlabeled_logits_tf: Tensor,
                             affine_transformer: t.Callable[[Tensor], Tensor],
                             unlabeled_image: Tensor, **kwargs):
        save_image_condition = self.epocher.cur_batch_num == 0 and self.epocher.cur_epoch % 3 == 0 and self.saver is not None

        n_unl = len(unlabeled_logits_tf)
        feature_ = self.extractor.feature()[-n_unl * 2:]
        _unlabeled_features, unlabeled_tf_features = torch.chunk(feature_, 2, dim=0)
        unlabeled_features_tf = affine_transformer(_unlabeled_features)

        feature_loss = self._run_feature_hooks(
            input1=unlabeled_features_tf,
            input2=unlabeled_tf_features,
            image=unlabeled_image_tf
        )
        if save_image_condition:
            self.saver.save_map(
                image=unlabeled_image_tf, feature_map1=unlabeled_tf_features, feature_map2=unlabeled_features_tf,
                cur_epoch=self.epocher.cur_epoch, cur_batch_num=self.epocher.cur_batch_num, save_name="feature"
            )

        projected_dist_tf, projected_tf_dist = zip(*[torch.chunk(x, 2) for x in self.projector(
            torch.cat([unlabeled_features_tf, unlabeled_tf_features], dim=0))])

        if save_image_condition:
            self.saver.save_map(
                image=unlabeled_image_tf, feature_map1=projected_dist_tf[0], feature_map2=projected_tf_dist[1],
                cur_epoch=self.epocher.cur_epoch, cur_batch_num=self.epocher.cur_batch_num, save_name="probability"
            )

        dist_losses = tuple(
            self._run_dist_hooks(
                input1=prob1, input2=prob2, image=unlabeled_image_tf,
                feature_map1=unlabeled_tf_features, feature_map2=unlabeled_features_tf, saver=self.saver,
                save_image_condition=save_image_condition, cur_epoch=self.epocher.cur_epoch,
                cur_batch_num=self.epocher.cur_batch_num
            )
            for prob1, prob2 in zip(projected_tf_dist, projected_dist_tf)
        )
        dist_loss = sum(dist_losses) / len(dist_losses)

        return feature_loss + dist_loss

    def _run_feature_hooks(self, **kwargs):
        return sum([h(**kwargs) for h in self._feature_hooks])

    def _run_dist_hooks(self, **kwargs):
        return sum([h(**kwargs) for h in self._dist_hooks])

    def close(self):
        self.extractor.remove()
        for h in chain(self._feature_hooks, self._dist_hooks):
            h.close()


class _CrossCorrelationHook(_TinyHook):

    def __init__(self, *, name: str = "cc", weight: float, kernel_size: int, diff_power: float = 0.75) -> None:
        criterion = CCLoss(win=(kernel_size, kernel_size))
        super().__init__(name=name, criterion=criterion, weight=weight)
        self._ent_func = Entropy(reduction="none")
        self._diff_power = diff_power

    def __call__(self, *, image: Tensor, input1: Tensor, input2: Tensor, saver: "FeatureMapSaver",
                 save_image_condition: bool, cur_epoch: int, cur_batch_num: int, **kwargs):
        if self.weight == 0:
            return torch.tensor(0.0, dtype=image.dtype, device=image.device)
        device = image.device
        self.criterion.to(device)  # noqa

        losses, self.diff_image, self.diff_prediction = zip(*[
            self.cc_loss_per_head(image=image, predict_simplex=x) for x in
            chain([input1, input2])
        ])
        loss = sum(losses) / len(losses)
        if self.meters:
            self.meters[self.name].add(loss.item())

        if save_image_condition:
            saver.save_map(
                image=self.diff_image[0], feature_map1=self.diff_prediction[0], feature_map2=self.diff_prediction[1],
                cur_epoch=cur_epoch, cur_batch_num=cur_batch_num,
                save_name="cross_correlation", feature_type="image"
            )

        return loss * self.weight

    def norm(self, image: Tensor, min=0.0, max=1.0, slicewise=True):
        if not slicewise:
            return self._norm(image, min, max)
        return torch.stack([self._norm(x) for x in image], dim=0)

    def _norm(self, image: Tensor, min=0.0, max=1.0):
        min_, max_ = image.min().detach(), image.max().detach()
        image = image - min_
        image = image / (max_ - min_ + 1e-6)
        return image * (max - min) + min

    @staticmethod
    def diff(image: Tensor):
        assert image.dim() == 4
        dx = image - torch.roll(image, shifts=1, dims=2)
        dy = image - torch.roll(image, shifts=1, dims=3)
        d = torch.sqrt(dx.pow(2) + dy.pow(2))
        return torch.mean(d, dim=1, keepdims=True)  # noqa

    def cc_loss_per_head(self, image: Tensor, predict_simplex: Tensor):
        if tuple(image.shape[-2:]) != tuple(predict_simplex.shape[-2:]):
            h, w = predict_simplex.shape[-2:]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                image = F.interpolate(image, size=(h, w), mode="bilinear")
        # the diff power applies only on edges.
        diff_image = self.norm(self.diff(image), min=0, max=1).pow(self._diff_power)
        diff_tf_softmax = self.norm(self._ent_func(predict_simplex), min=0, max=1, slicewise=False).unsqueeze(1)

        loss = self.criterion(
            diff_tf_softmax,
            diff_image
        )
        return loss, diff_image, diff_tf_softmax


class _MIHook(_TinyHook):

    def __init__(self, *, name: str = "mi", weight: float, lamda: float, padding: int = 0) -> None:
        criterion = IIDSegmentationLoss(lamda=lamda, padding=padding)
        super().__init__(name=name, criterion=criterion, weight=weight)

    def __call__(self, input1: Tensor, input2: Tensor, **kwargs):
        if self.weight == 0:
            return torch.tensor(0, device=input1.device, dtype=input1.dtype)
        loss = self.criterion(input1, input2)
        if self.meters:
            self.meters[self.name].add(loss.item())
        return loss * self.weight


class _CenterCompactnessHook(_TinyHook):

    def __init__(self, *, name: str = "center", weight: float) -> None:
        super().__init__(name=name, criterion=None, weight=weight)  # noqa
        del self.criterion

    def __call__(self, input1, input2, feature_map1, feature_map2, image, **kwargs):
        if self.weight == 0:
            return torch.tensor(0, dtype=input1.dtype, device=input1.device)
        if torch.rand((1,)).item() > 0.5:
            return torch.tensor(0, dtype=input1.dtype, device=input1.device)

        if torch.rand((1,)).item() > 0.5:
            loss = self.forward(probability_simplex=input1, feature_map=feature_map1)
        else:
            loss = self.forward(probability_simplex=input2, feature_map=feature_map2)
        if self.meters:
            self.meters[self.name].add(loss.item())
        return loss * self.weight

    def forward(self, probability_simplex: Tensor, feature_map: Tensor):
        dims = probability_simplex.shape[1]
        one_hot_mask = probs2one_hot(probability_simplex, class_dim=1)
        losses = []
        # randomly choose some dimension
        for dim in random.sample(range(dims), max(5, dims // 5)):
            cur_mask = one_hot_mask[:, dim].bool().unsqueeze(1)
            if cur_mask.sum() == 0:
                continue
            prototype = self.masked_average_pooling2(feature_map, cur_mask)
            loss = self.center_loss(feature=feature_map, mask=cur_mask, prototype=prototype)
            losses.append(loss)
        if len(losses) == 0:
            return torch.tensor(0, dtype=probability_simplex.dtype, device=probability_simplex.device)

        return sum(losses) / len(losses)

    def masked_average_pooling(self, feature: Tensor, mask: Tensor):
        assert feature.dim() == 4
        data = feature.masked_fill(mask == 0, 0)
        nominator = torch.sum(data, dim=(0, 2, 3), keepdim=True)
        denominator = torch.sum(mask.type(nominator.dtype), dim=(0, 2, 3), keepdim=True)
        return nominator / (denominator + 1e-16)

    def masked_average_pooling2(self, feature: Tensor, mask: Tensor):
        b, c, *_ = feature.shape
        used_feature = feature.swapaxes(1, 0).masked_select(mask.swapaxes(1, 0)).reshape(c, -1)
        return used_feature.mean(dim=1).reshape(1, c, 1, 1)

    def center_loss(self, feature: Tensor, mask: Tensor, prototype: Tensor):
        return torch.mean((feature - prototype).pow(2), dim=1, keepdim=True).masked_select(mask).mean()
