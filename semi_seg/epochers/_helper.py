import random
from contextlib import contextmanager
from typing import Callable
from typing import Union, List

import torch
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.meters2 import EpochResultDict, MeterInterface
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.tqdm import tqdm, item2str
from loguru import logger
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer

from contrastyou.featextractor.unet import FeatureExtractorWithIndex as FeatureExtractor


class unl_extractor:
    def __init__(self, features: FeatureExtractor, n_uls: int) -> None:
        super().__init__()
        self._features = features
        self._n_uls = n_uls

    def __iter__(self):
        for feature in self._features:
            assert len(feature) >= self._n_uls, (len(feature), self._n_uls)
            yield feature[len(feature) - self._n_uls:]


@contextmanager
def set_grad_tensor(tensor: Tensor, is_enable: bool):
    prev_flag = tensor.requires_grad
    tensor.requires_grad = is_enable
    yield
    tensor.requires_grad = prev_flag


@contextmanager
def set_grad_module(module: nn.Module, is_enable: bool):
    prev_flags = {k: v.requires_grad for k, v in module.named_parameters()}
    for k, v in module.named_parameters():
        v.requires_grad = is_enable
    yield
    for k, v in module.named_parameters():
        v.requires_grad = prev_flags[k]


def set_grad(tensor_or_module: Union[Tensor, nn.Module], is_enable):
    assert isinstance(tensor_or_module, (Tensor, nn.Module))
    if isinstance(tensor_or_module, Tensor):
        return set_grad_tensor(tensor_or_module, is_enable)
    return set_grad_module(tensor_or_module, is_enable)


class __AssertWithUnLabeledData:

    def _assertion(self):
        from .base import FineTuneEpocher
        assert not isinstance(self, FineTuneEpocher)
        logger.debug("{} using unlabeled data checked!", self.__class__.__name__)


class __AssertOnlyWithLabeledData:

    def _assertion(self):
        from .base import FineTuneEpocher
        assert isinstance(self, FineTuneEpocher)
        logger.debug("{} using only labeled data checked!", self.__class__.__name__)


# feature extractor mixin
class _FeatureExtractorMixin:

    def __init__(self, feature_position: Union[List[str], str], feature_importance: Union[float, List[float]], *args,
                 **kwargs):
        super(_FeatureExtractorMixin, self).__init__(*args, **kwargs)

        assert isinstance(feature_position, list) and isinstance(feature_position[0], str), feature_position
        assert isinstance(feature_importance, list) and isinstance(feature_importance[0],
                                                                   (int, float)), feature_importance
        self._feature_position = feature_position
        self._feature_importance = feature_importance
        logger.debug("Initializing {} with features and weights: {}", self.__class__.__name__,
                     item2str({k: v for k, v in zip(self._feature_position, self._feature_importance)}))

        assert len(self._feature_position) == len(self._feature_importance), \
            (len(self._feature_position), len(self._feature_importance))

    def run(self, *args, **kwargs):
        with FeatureExtractor(self._model, self._feature_position) as self._fextractor:  # noqa
            logger.debug(f"create feature extractor for {', '.join(self._feature_position)} ")
            return super(_FeatureExtractorMixin, self).run(*args, **kwargs)  # noqa

    def forward_pass(self, *args, **kwargs):
        self._fextractor.clear()
        with self._fextractor.enable_register():
            return super(_FeatureExtractorMixin, self).forward_pass(*args, **kwargs)  # noqa


# ======== base pretrain epocher mixin ================
class _PretrainEpocherMixin:
    """
    PretrainEpocher makes all images goes to regularization, permitting to use the other classes to create more pretrain
    models
    """
    meters: MeterInterface
    _model: nn.Module
    _optimizer: Optimizer
    _indicator: tqdm
    _unzip_data: Callable[..., torch.device]
    _device: torch.device
    _affine_transformer: Callable[[Tensor], Tensor]
    on_master: Callable[[], bool]
    regularization: Callable[..., Tensor]
    forward_pass: Callable

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meter = super()._configure_meters(meters)  # noqa
        meter.delete_meters(["sup_loss", "sup_dice", "reg_weight"])
        return meter

    def init(self, *, chain_dataloader, **kwargs):
        # extend the interface for original class with chain_dataloader
        super().init(**kwargs)  # noqa
        self._chain_dataloader = chain_dataloader

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        assert self._model.training, self._model.training
        return self._run_pretrain(*args, **kwargs)

    def _run_pretrain(self, *args, **kwargs):
        for i, data in zip(self._indicator, self._chain_dataloader):
            seed = random.randint(0, int(1e7))
            unlabeled_image, unlabeled_target, unlabeled_filename, unl_partition, unl_group = \
                self._unzip_data(data, self._device)

            with FixRandomSeed(seed):
                unlabeled_image_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_image], dim=0)
            assert unlabeled_image_tf.shape == unlabeled_image.shape, \
                (unlabeled_image_tf.shape, unlabeled_image.shape)

            unlabel_logits, unlabel_tf_logits = self.forward_pass(
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf
            )

            with FixRandomSeed(seed):
                unlabel_logits_tf = torch.stack([self._affine_transformer(x) for x in unlabel_logits], dim=0)

            assert unlabel_logits_tf.shape == unlabel_tf_logits.shape, (
                unlabel_logits_tf.shape, unlabel_tf_logits.shape)

            # regularized part
            reg_loss = self.regularization(
                unlabeled_tf_logits=unlabel_tf_logits,
                unlabeled_logits_tf=unlabel_logits_tf,
                seed=seed,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf,
                label_group=unl_group,
                partition_group=unl_partition,
                unlabeled_filename=unlabeled_filename,
            )
            total_loss = reg_loss
            # gradient backpropagation
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()
            # recording can be here or in the regularization method
            if self.on_master():
                with torch.no_grad():
                    self.meters["reg_loss"].add(reg_loss.item())
                    report_dict = self.meters.tracking_status()
                    self._indicator.set_postfix_dict(report_dict)

        report_dict = self.meters.tracking_status(final=True)
        return report_dict

    def _forward_pass(self, unlabeled_image, unlabeled_image_tf):
        n_l, n_unl = 0, len(unlabeled_image)
        predict_logits = self._model(torch.cat([unlabeled_image, unlabeled_image_tf], dim=0))

        unlabel_logits, unlabel_tf_logits = torch.split(predict_logits, [n_unl, n_unl], dim=0)
        return unlabel_logits, unlabel_tf_logits
