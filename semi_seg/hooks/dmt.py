from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from typing import Sequence

import torch
from loguru import logger
from torch import nn, Tensor

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.losses.dice_loss import DiceLoss
from contrastyou.losses.kl import KL_div
from contrastyou.meters import AverageValueMeter, MeterInterface
from contrastyou.utils import class2one_hot
from semi_seg.hooks import meter_focus
from semi_seg.hooks.mt import L2LossChecker, EMAUpdater


@contextmanager
def manually_forward(network: nn.Module, gradient: Sequence[Tensor], lambda_: float):
    for g, p in zip(gradient, network.parameters()):
        p.data.sub_(g, alpha=lambda_)
    yield
    for g, p in zip(gradient, network.parameters()):
        p.data.add_(g, alpha=lambda_)


@contextmanager
def set_model_status(model: nn.Module, *, training: bool):
    previous_state = model.training
    model.train(training)
    yield
    model.train(previous_state)


class DifferentiableMeanTeacherTrainerHook(TrainerHook):

    def __init__(self, *, name: str, model: nn.Module, weight: float, alpha: float = 0.999, weight_decay: float = 1e-5,
                 meta_weight=1e-3, meta_criterion: str):
        super().__init__(hook_name=name)
        self._weight = weight
        self._criterion = nn.MSELoss()
        assert meta_criterion in ("ce", "dice")
        self._meta_criterion = KL_div() if meta_criterion == "ce" else DiceLoss(ignore_index=0)
        self._updater = EMAUpdater(alpha=alpha, weight_decay=weight_decay)
        self._teacher_model = deepcopy(model)
        self._meta_weight = meta_weight
        logger.opt(depth=0).trace("meta_weight: {}".format(meta_weight))

    def __call__(self):
        return _DifferentiableMeanTeacherEpocherHook(
            name=self._hook_name, weight=self._weight, student_criterion=self._criterion,
            teacher_criterion=self._meta_criterion,
            teacher_model=self.teacher_model, updater=self._updater, meta_weight=self._meta_weight
        )

    @property
    def teacher_model(self):
        return self._teacher_model


class _DifferentiableMeanTeacherEpocherHook(EpocherHook):
    def __init__(self, name: str, weight: float, student_criterion, teacher_criterion, teacher_model: nn.Module,
                 meta_weight: float, updater: EMAUpdater) -> None:
        super().__init__(name)
        self._weight = weight
        self._student_criterion = L2LossChecker(student_criterion)
        self._teacher_criterion = L2LossChecker(teacher_criterion)
        self._meta_weight: float = meta_weight
        self._teacher_model = teacher_model
        self._updater = updater

        self._teacher_model.train()

    @meter_focus
    def configure_meters(self, meters: MeterInterface):
        self.meters.register_meter("consistency_loss", AverageValueMeter())
        self.meters.register_meter("teacher_loss", AverageValueMeter())

    @meter_focus
    def __call__(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
                 labeled_image, labeled_target, **kwargs):
        teacher_model = self.teacher_model

        with set_model_status(teacher_model, training=False):
            teacher_labeled_prob = teacher_model(labeled_image).softmax(1)
            one_hot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes).float()

        teacher_loss = self._teacher_criterion(teacher_labeled_prob, one_hot_target)
        _grad_teacher = torch.autograd.grad(teacher_loss, tuple(teacher_model.parameters()), only_inputs=True)

        with manually_forward(teacher_model, _grad_teacher, lambda_=self._meta_weight):
            # manually get the model t+1 and reset to t when exiting
            student_unlabeled_tf_prob = unlabeled_tf_logits.softmax(1)
            with torch.no_grad():
                teacher_unlabeled_prob = self._teacher_model(unlabeled_image).softmax(1)
                teacher_unlabeled_prob_tf = affine_transformer(teacher_unlabeled_prob)
        loss = self._student_criterion(teacher_unlabeled_prob_tf, student_unlabeled_tf_prob)
        self.meters["consistency_loss"].add(loss.item())
        self.meters["teacher_loss"].add(teacher_loss.item())

        return self._weight * loss

    def after_batch_update(self, **kwargs):
        self._updater(ema_model=self._teacher_model, student_model=self.model)

    @property
    @lru_cache()
    def num_classes(self):
        return self._epocher.num_classes  # noqa

    @property
    def model(self):
        return self.epocher._model  # noqa

    @property
    def teacher_model(self):
        return self._teacher_model
