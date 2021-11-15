import typing as t
import warnings
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
from contrastyou.utils import class_name, average_iter, item2str
from semi_seg.hooks.utils import FeatureMapSaver

if t.TYPE_CHECKING:
    from contrastyou.projectors.nn import _ProjectorHeadBase  # noqa
    from contrastyou.meters import MeterInterface

__all__ = ["CrossCorrelationHook", "CrossCorrelationHookWithSaver"]


class CrossCorrelationHook(TrainerHook):

    def __init__(self, *, name: str, model: nn.Module, feature_name: UNetFeatureMapEnum, cc_weight: float,
                 mi_weight: float = 0.0, kernel_size: int, projector_params: t.Dict[str, t.Any],
                 mi_criterion_params: t.Dict[str, t.Any], norm_params: t.Dict[str, t.Any]):
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
        input_dim = model.get_channel_dim(feature_name.value)  # model: type: UNet
        logger.trace(f"Creating projector with {item2str(projector_params)}")
        self._projector = CrossCorrelationProjector(input_dim=input_dim, **projector_params)

        logger.trace(f"Creating CCLoss with kernel_size = {kernel_size} with weight = {self._cc_weight}.")
        self._cc_criterion = CCLoss(win=(kernel_size, kernel_size))

        logger.trace(f"Creating IIDSegmentationLoss with kernel_size = {kernel_size} with weight = {self._mi_weight}.")
        self._mi_criterion = IIDSegmentationLoss(**mi_criterion_params)

        self._diff_power = float(norm_params["power"])

    def __call__(self, **kwargs):
        return _CrossCorrelationEpocherHook(
            name=self._hook_name, extractor=self._extractor,
            projector=self._projector, cc_criterion=self._cc_criterion,
            cc_weight=self._cc_weight, mi_weight=self._mi_weight,
            mi_criterion=self._mi_criterion, diff_power=self.diff_power
        )

    @property
    def learnable_modules(self) -> t.List[nn.Module]:
        return [self._projector, ]


class _CrossCorrelationEpocherHook(EpocherHook):

    def __init__(self, *, name: str = "cc", extractor: 'SingleFeatureExtractor', projector: '_ProjectorHeadBase',
                 cc_criterion: CCLoss, mi_criterion: 'IIDSegmentationLoss',
                 cc_weight: float, mi_weight: float, diff_power: float = 1.0) -> None:
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
        self, unlabeled_image_tf: Tensor, unlabeled_logits_tf: Tensor, affine_transformer: t.Callable[[Tensor], Tensor],
        unlabeled_image: Tensor, **kwargs
    ):
        n_unl = len(unlabeled_logits_tf)
        feature_ = self.extractor.feature()[-n_unl * 2:]
        _unlabeled_features, unlabeled_tf_features = torch.chunk(feature_, 2, dim=0)
        unlabeled_features_tf = affine_transformer(_unlabeled_features)

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

        diff_image = self.norm(self.diff(image), min=0, max=1).pow(self._diff_power)
        diff_tf_softmax = self.norm(self._ent_func(predict_simplex), min=0, max=1, slicewise=False).unsqueeze(1)

        loss = self.cc_criterion(
            diff_tf_softmax,
            diff_image
        )
        return loss, diff_image, diff_tf_softmax

    def mi_loss_per_head(self, prob1, prob2):
        loss = sum([self.mi_criterion(x1, x2) for x1, x2 in zip(prob1, prob2)]) / len(prob1)
        return loss


class CrossCorrelationHookWithSaver(CrossCorrelationHook):
    # with an image saver

    def __init__(self, *, name: str, model: nn.Module, feature_name: UNetFeatureMapEnum, cc_weight: float,
                 mi_weight: float = 0.0, kernel_size: int, projector_params: t.Dict[str, t.Any],
                 mi_criterion_params: t.Dict[str, t.Any], norm_params: t.Dict[str, t.Any], save: bool = False):
        super().__init__(
            name=name, model=model, feature_name=feature_name, cc_weight=cc_weight, mi_weight=mi_weight,
            kernel_size=kernel_size, projector_params=projector_params, mi_criterion_params=mi_criterion_params,
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
            diff_power=self._diff_power
        )

    def close(self):
        if self.saver:
            self.saver.zip()


class _CrossCorrelationEpocherHookWithSaver(_CrossCorrelationEpocherHook):
    # with an image saver

    def __init__(self, *, name: str = "cc", extractor: 'SingleFeatureExtractor', projector: '_ProjectorHeadBase',
                 cc_criterion: 'CCLoss', mi_criterion: 'IIDSegmentationLoss', cc_weight: float,
                 mi_weight: float, diff_power: float, saver: 'FeatureMapSaver') -> None:
        super().__init__(name=name, extractor=extractor, projector=projector, cc_criterion=cc_criterion,
                         mi_criterion=mi_criterion, cc_weight=cc_weight, mi_weight=mi_weight, diff_power=diff_power)
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
