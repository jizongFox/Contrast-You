# this hook tries to use patch-based Cross Correlation loss on the over-segmentation softmax and the original image.
import typing as t
import warnings
from itertools import chain

import torch
from loguru import logger
from torch import Tensor
from torch.nn import functional as F

from contrastyou.arch import UNetFeatureMapEnum
from contrastyou.hooks import TrainerHook, EpocherHook
from contrastyou.losses.cross_correlation import CCLoss
from contrastyou.losses.discreteMI import IIDSegmentationLoss
from contrastyou.losses.kl import Entropy
from contrastyou.meters import MeterInterface, AverageValueMeter
from contrastyou.utils import average_iter, class_name
from semi_seg.hooks.utils import FeatureMapSaver


class CrossCorrelationOnLogitsHook(TrainerHook):

    def __init__(self, *, name: str, feature_name: UNetFeatureMapEnum, cc_weight: float,
                 mi_weight: float = 0.0, kernel_size: int,
                 mi_criterion_params: t.Dict[str, t.Any], norm_params: t.Dict[str, t.Any], save: bool = True, **kwargs):
        super().__init__(hook_name=name)
        self._cc_weight = float(cc_weight)
        self._mi_weight = float(mi_weight)
        feature_name = UNetFeatureMapEnum(feature_name)
        logger.info(
            f"Creating {class_name(self)} @{feature_name.name}.")
        self._feature_name = feature_name.value
        assert feature_name == UNetFeatureMapEnum.Deconv_1x1

        logger.trace(f"Creating CCLoss with kernel_size = {kernel_size} with weight = {self._cc_weight}.")
        self._cc_criterion = CCLoss(win=(kernel_size, kernel_size))

        logger.trace(f"Creating IIDSegmentationLoss with kernel_size = {kernel_size} with weight = {self._mi_weight}.")
        self._mi_criterion = IIDSegmentationLoss(**mi_criterion_params)

        self._diff_power: float = float(norm_params["power"])
        assert 0 <= self._diff_power <= 1, self._diff_power

        self.save = save
        self.saver = None

    def after_initialize(self):
        if self.save:
            self.saver = FeatureMapSaver(save_dir=self.trainer.absolute_save_dir, folder_name="vis/logit")

    def __call__(self, **kwargs):
        return _CrossCorrelationLogitEpocherHook(
            name=self._hook_name, cc_criterion=self._cc_criterion, mi_criterion=self._mi_criterion,
            cc_weight=self._cc_weight, mi_weight=self._mi_weight, diff_power=self._diff_power, saver=self.saver,
        )

    def close(self):
        if self.saver:
            self.saver.zip()


class _CrossCorrelationLogitEpocherHook(EpocherHook):

    def __init__(self, *, name: str = "cc", cc_criterion: 'CCLoss', mi_criterion: 'IIDSegmentationLoss',
                 cc_weight: float,
                 mi_weight: float, diff_power: float, saver: 'FeatureMapSaver') -> None:
        super().__init__(name=name)
        self.cc_weight = cc_weight
        self.mi_weight = mi_weight
        self.cc_criterion = cc_criterion
        self.mi_criterion = mi_criterion
        self._ent_func = Entropy(reduction="none")
        self._diff_power = diff_power
        self.saver = saver

    def configure_meters_given_epocher(self, meters: 'MeterInterface'):
        meters.register_meter("cc_ls", AverageValueMeter())
        meters.register_meter("mi_ls", AverageValueMeter())
        return meters

    def _call_implementation(self, unlabeled_image_tf: Tensor, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor,
                             **kwargs):
        save_image_condition = self.epocher.cur_batch_num == 0 and self.epocher.cur_epoch % 5 == 0
        if save_image_condition:
            self.saver.save_map(
                image=unlabeled_image_tf, feature_map1=unlabeled_tf_logits, feature_map2=unlabeled_logits_tf,
                cur_epoch=self.epocher.cur_epoch, cur_batch_num=self.epocher.cur_batch_num, save_name="logits"
            )

        losses, diff_image, diff_prediction = zip(*[
            self.cc_loss_per_head(image=unlabeled_image_tf, predict_simplex=x) for x in
            chain([unlabeled_logits_tf.softmax(1), unlabeled_tf_logits.softmax(1)])
        ])

        if save_image_condition:
            self.saver.save_map(
                image=diff_image[0], feature_map1=diff_prediction[0], feature_map2=diff_prediction[0],
                cur_epoch=self.epocher.cur_epoch, cur_batch_num=self.epocher.cur_batch_num,
                save_name="cross_correlation", feature_type="image"
            )
        cc_loss = average_iter(losses)
        mi_loss = self.mi_loss_per_head(unlabeled_logits_tf.softmax(1), unlabeled_tf_logits.softmax(1))
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

        diff_image = self.norm(self.diff(image), min=0, max=1).pow(
            self._diff_power)  # the diff power applies only on edges.
        diff_tf_softmax = self.norm(self._ent_func(predict_simplex), min=0, max=1, slicewise=False).unsqueeze(1)

        loss = self.cc_criterion(
            diff_tf_softmax,
            diff_image
        )
        return loss, diff_image, diff_tf_softmax

    def mi_loss_per_head(self, prob1, prob2):
        loss = self.mi_criterion(prob1, prob2)
        return loss
