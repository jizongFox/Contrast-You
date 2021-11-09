# this hook tries to use patch-based Cross Correlation loss on the over-segmentation softmax and the original image.
import typing
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

if typing.TYPE_CHECKING:
    from contrastyou.projectors.nn import _ProjectorHeadBase  # noqa
    from contrastyou.meters import MeterInterface

__all__ = ["CrossCorrelationHook"]


class CrossCorrelationHook(TrainerHook):

    def __init__(self, *, name: str, model: nn.Module, feature_name: UNetFeatureMapEnum, cc_weight: float,
                 mi_weight: float = 0.0, kernel_size: int, projector_params: t.Dict[str, t.Any]):
        super().__init__(hook_name=name)
        self._cc_weight = cc_weight
        self._mi_weight = mi_weight
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
        self._mi_criterion = IIDSegmentationLoss(padding=0)

    def __call__(self, **kwargs):
        return _CrossCorrelationEpocherHook(
            name=self._hook_name, extractor=self._extractor,
            projector=self._projector, cc_criterion=self._cc_criterion,
            cc_weight=self._cc_weight, mi_weight=self._mi_weight,
            mi_criterion=self._mi_criterion
        )

    @property
    def learnable_modules(self) -> t.List[nn.Module]:
        return [self._projector, ]


class _CrossCorrelationEpocherHook(EpocherHook):

    def __init__(self, *, name: str = "cc", extractor: 'SingleFeatureExtractor', projector: '_ProjectorHeadBase',
                 cc_criterion: CCLoss, mi_criterion: 'IIDSegmentationLoss',
                 cc_weight: float, mi_weight: float) -> None:
        super().__init__(name=name)
        self.cc_weight = cc_weight
        self.mi_weight = mi_weight
        self.extractor = extractor
        self.extractor.bind()
        self.projector = projector
        self.cc_criterion = cc_criterion
        self.mi_criterion = mi_criterion
        self._ent_func = Entropy(reduction="none")

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

    def _call_implementation(self, unlabeled_image_tf: Tensor, unlabeled_logits_tf: Tensor,
                             affine_transformer: t.Callable[[Tensor], Tensor], unlabeled_image: Tensor,
                             **kwargs):
        n_unl = len(unlabeled_logits_tf)
        feature_ = self.extractor.feature()[-n_unl * 2:]
        _unlabeled_features, unlabeled_tf_features = torch.chunk(feature_, 2, dim=0)
        unlabeled_features_tf = affine_transformer(_unlabeled_features)

        projected_dist_tf, projected_tf_dist = zip(*[torch.chunk(x, 2) for x in self.projector(
            torch.cat([unlabeled_features_tf, unlabeled_tf_features], dim=0))])

        def cc_loss_per_head(image: Tensor, predict_simplex: Tensor):
            # resize_image
            if tuple(image.shape) != tuple(predict_simplex.shape):
                h, w = predict_simplex.shape[-2:]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    image = F.interpolate(image, size=(h, w), mode="bilinear")

            diff_image = self.diff(image)
            diff_tf_softmax = self._ent_func(predict_simplex).unsqueeze(1)

            loss = self.cc_criterion(
                self.norm(diff_tf_softmax),
                self.norm(diff_image)
            )
            return loss

        def mi_loss_per_head(prob1, prob2):
            loss = sum([self.mi_criterion(x1, x2) for x1, x2 in zip(prob1, prob2)]) / len(prob1)
            return loss

        losses = [
            cc_loss_per_head(image=unlabeled_image_tf, predict_simplex=x) for x in
            chain(projected_dist_tf, projected_tf_dist)
        ]
        cc_loss = average_iter(losses)
        mi_loss = mi_loss_per_head(projected_dist_tf, projected_tf_dist)
        if self.meters:
            self.meters["cc_ls"].add(cc_loss.item())
            self.meters["mi_ls"].add(mi_loss.item())
        return cc_loss * self.cc_weight + mi_loss * self.mi_weight

    @staticmethod
    def norm(image: Tensor):
        min_, max_ = image.min().detach(), image.max().detach()
        image = image - min_
        image = image / (max_ - min_ + 1e-6)
        return (image - 0.5) * 2

    @staticmethod
    def diff(image: Tensor):
        assert image.dim() == 4
        dx = image - torch.roll(image, shifts=1, dims=2)
        dy = image - torch.roll(image, shifts=1, dims=3)
        d = torch.sqrt(dx.pow(2) + dy.pow(2))
        return torch.mean(d, dim=[1], keepdims=True)  # noqa
