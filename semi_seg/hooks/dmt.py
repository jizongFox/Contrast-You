from contextlib import contextmanager
from copy import deepcopy
from typing import Sequence, List

import torch
from loguru import logger
from torch import nn, Tensor
from torch.cuda.amp import GradScaler

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.losses.dice_loss import DiceLoss
from contrastyou.losses.kl import KL_div
from contrastyou.meters import AverageValueMeter, MeterInterface
from contrastyou.types import CriterionType
from contrastyou.utils import class2one_hot, class_name
from semi_seg.hooks import meter_focus
from semi_seg.hooks.mt import L2LossChecker, EMAUpdater


@contextmanager
def manually_forward_with_grad(module: nn.Module, grad: Sequence[Tensor], lambda_: float):
    for g, p in zip(grad, module.parameters()):
        p.data.sub_(g, alpha=lambda_)
    yield
    for g, p in zip(grad, module.parameters()):
        p.data.add_(g, alpha=lambda_)


@contextmanager
def manually_forward_with_ema(teacher_module: nn.Module, student_module: nn.Module, *, alpha: float):
    update_func = EMAUpdater(justify_alpha=False, alpha=alpha)
    teacher_ckpt = deepcopy(teacher_module.state_dict())
    update_func(ema_model=teacher_module, student_model=student_module)
    yield
    teacher_module.load_state_dict(teacher_ckpt)


@contextmanager
def switch_model_status(model: nn.Module, *, training: bool):
    previous_state = model.training
    model.train(training)
    yield
    model.train(previous_state)


class DifferentiableMeanTeacherTrainerHook(TrainerHook):

    def __init__(self, *, name: str, model: nn.Module, weight: float, alpha: float = 0.999, weight_decay: float = 1e-5,
                 meta_weight=1e-3, meta_criterion: str, method_name: str):
        super().__init__(hook_name=name)
        self._weight = weight
        self._criterion = nn.MSELoss()
        assert meta_criterion in ("ce", "dice")
        self._meta_criterion = KL_div() if meta_criterion == "ce" else DiceLoss(ignore_index=0)
        self._updater = EMAUpdater(alpha=alpha, weight_decay=weight_decay)
        self._teacher_model = deepcopy(model)
        self._meta_weight = meta_weight
        logger.opt(depth=0).trace("meta_weight: {}".format(meta_weight))
        self._method_name = method_name
        assert method_name in ("method1", "method2"), method_name
        logger.info(f"{class_name(self)} method name: {method_name}")

    def __call__(self):
        hook_name = _DifferentiableMeanTeacherEpocherHook2 if self._method_name == "method2" else \
            _DifferentiableMeanTeacherEpocherHook

        return hook_name(
            name=self._hook_name, weight=self._weight, student_criterion=self._criterion,
            meta_criterion=self._meta_criterion, teacher_model=self.teacher_model, updater=self._updater,
            meta_weight=self._meta_weight)

    @property
    def teacher_model(self):
        return self._teacher_model

    @property
    def learnable_modules(self) -> List[nn.Module]:
        return [self._teacher_model]


class _BaseDMTEpocherHook(EpocherHook):
    """
    Base MT Epocher
    """

    def __init__(self, *, name: str, model: nn.Module, teacher_model: nn.Module, student_criterion: CriterionType,
                 meta_criterion: CriterionType, weight: float, meta_weight: float, updater: EMAUpdater) -> None:
        super().__init__(name)
        self._model = model
        self._teacher_model = teacher_model

        self._weight = weight
        self._meta_weight: float = meta_weight

        self._criterion = L2LossChecker(student_criterion)
        self._meta_criterion = L2LossChecker(meta_criterion)

        self._updater = updater
        self._teacher_model.train()
        self._model.train()

    def mt_update(self, *, unlabeled_tf_logits, unlabeled_image, affine_transformer):
        # taken from mean teacher
        student_unlabeled_tf_prob = unlabeled_tf_logits.softmax(1)
        with torch.no_grad():
            teacher_unlabeled_prob = self.teacher_model(unlabeled_image).softmax(1)
            teacher_unlabeled_prob_tf = affine_transformer(teacher_unlabeled_prob)
        loss = self._criterion(teacher_unlabeled_prob_tf, student_unlabeled_tf_prob)
        return loss

    @property
    def num_classes(self):
        return self._epocher.num_classes

    @property
    def model(self):
        return self._model

    @property
    def teacher_model(self):
        return self._teacher_model

    @property
    def optimizer(self):
        return self.epocher._optimizer  # noqa

    @property
    def scaler(self) -> GradScaler:
        return self.epocher.scaler

    @property
    def cur_batch_num(self):
        return self.epocher.cur_batch_num


class _MTEpocherHook(_BaseDMTEpocherHook):

    @meter_focus
    def configure_meters(self, meters: MeterInterface):
        self.meters.register_meter("loss", AverageValueMeter())

    @meter_focus
    def __call__(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
                 **kwargs):
        student_unlabeled_tf_prob = unlabeled_tf_logits.softmax(1)
        with torch.no_grad():
            teacher_unlabeled_prob = self.teacher_model(unlabeled_image).softmax(1)
            teacher_unlabeled_prob_tf = affine_transformer(teacher_unlabeled_prob)
        loss = self._criterion(teacher_unlabeled_prob_tf, student_unlabeled_tf_prob)
        self.meters["loss"].add(loss.item())
        return self._weight * loss

    def after_batch_update(self, **kwargs):
        # using previous teacher to update the next teacher.
        self._updater(ema_model=self.teacher_model, student_model=self.model)


class SecondDifferentiableMeanTeacherEpocherHook(_BaseDMTEpocherHook):
    """Maybe I need to add this as an epocher instead of hook"""

    def before_batch_update(self, **kwargs):
        if self.cur_batch_num == 0:
            self.epocher.retain_graph = True

    @meter_focus
    def __call__(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
                 labeled_image, labeled_target, **kwargs):
        # student t checkpoint

        loss = self.mt_update(unlabeled_tf_logits=unlabeled_tf_logits, unlabeled_image=unlabeled_image,
                              affine_transformer=affine_transformer)
        self.meters["consistency_loss"].add(loss.item())

        # saving checkpoints for S^(t)
        self._student_ckpt = deepcopy(self.model.state_dict())
        self._teacher_ckpt = deepcopy(self.teacher_model.state_dict())
        self._optimizer_ckpt = deepcopy(self.optimizer.state_dict())
        self._scaler_ckpt = deepcopy(self.scaler.state_dict())
        return self._weight * loss

    @meter_focus
    def after_batch_update(self, labeled_image, labeled_target, seed, **kwargs):
        # after batch backward, student becomes S^(t+1)
        # getting a new teacher T^(t+1)
        self._updater(ema_model=self.teacher_model, student_model=self.model)

        self.teacher_model.zero_grad()
        with switch_model_status(self.teacher_model, training=False):
            meta_pred = self.teacher_model(labeled_image).softmax(1)
        meta_oh_target = class2one_hot(labeled_target.squeeze(1), C=self.num_classes).float()
        meta_loss = self._meta_criterion(meta_pred, meta_oh_target)
        meta_grad = torch.autograd.grad(meta_loss, tuple(self.teacher_model.parameters()), only_inputs=True)

        sec_student_grad = torch.autograd.grad()
        student_grad = tuple([x.grad.clone() for x in self.model.parameters()])

        self.meters["teacher_loss"].add(meta_loss.item())

        self.model.zero_grad()
        self.teacher_model.load_state_dict(self.__teacher_ckpt)
        self.model.load_state_dict(self.__student_ckpt)
        self.optimizer.load_state_dict(self.__optimizer_ckpt)

        # with autocast(enabled=False):
        # update the weighted gradient.
        with torch.no_grad():
            for i, (meta_g, student_g, p) in enumerate(zip(meta_grad, student_grad, self.model.parameters())):
                p.grad.data.copy_(student_g.data + self._meta_weight * meta_g.data)

        self.scaler.step(self.optimizer)
        self._updater(ema_model=self.teacher_model, student_model=self.model)
        self.optimizer.zero_grad()


class _DifferentiableMeanTeacherEpocherHook(_BaseDMTEpocherHook):

    @meter_focus
    def configure_meters(self, meters: MeterInterface):
        self.meters.register_meter("consistency_loss", AverageValueMeter())
        self.meters.register_meter("teacher_loss", AverageValueMeter())

    @meter_focus
    def __call__(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
                 labeled_image, labeled_target, **kwargs):
        teacher_model = self.teacher_model

        with switch_model_status(teacher_model, training=False):
            teacher_labeled_prob = teacher_model(labeled_image).softmax(1)
        one_hot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes).float()

        teacher_loss = self._meta_criterion(teacher_labeled_prob, one_hot_target)
        _grad_teacher = torch.autograd.grad(teacher_loss, tuple(teacher_model.parameters()), only_inputs=True)

        with manually_forward_with_grad(teacher_model, _grad_teacher, lambda_=self._meta_weight):
            # manually get the model t+1 and reset to t when exiting
            student_unlabeled_tf_prob = unlabeled_tf_logits.softmax(1)
            with torch.no_grad():
                teacher_unlabeled_prob = self._teacher_model(unlabeled_image).softmax(1)
            teacher_unlabeled_prob_tf = affine_transformer(teacher_unlabeled_prob)
        loss = self._criterion(teacher_unlabeled_prob_tf, student_unlabeled_tf_prob)
        self.meters["consistency_loss"].add(loss.item())
        self.meters["teacher_loss"].add(teacher_loss.item())

        return self._weight * loss

    def after_batch_update(self, **kwargs):
        self._updater(ema_model=self._teacher_model, student_model=self.model)


class _DifferentiableMeanTeacherEpocherHook2(_DifferentiableMeanTeacherEpocherHook):

    @meter_focus
    def __call__(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
                 labeled_image, labeled_target, **kwargs):
        # student t checkpoint

        loss = self.mt_update(unlabeled_tf_logits=unlabeled_tf_logits, unlabeled_image=unlabeled_image,
                              affine_transformer=affine_transformer)
        self.meters["consistency_loss"].add(loss.item())

        self._student_ckpt = deepcopy(self.model.state_dict())
        self._teacher_ckpt = deepcopy(self.teacher_model.state_dict())
        self._optimizer_ckpt = deepcopy(self.optimizer.state_dict())

        return self._weight * loss

    @meter_focus
    def after_batch_update(self, labeled_image, labeled_target, seed, **kwargs):
        # getting a new teacher
        self._updater(ema_model=self.teacher_model, student_model=self.model)
        # backup student gradient
        student_grad = tuple([x.grad.clone() for x in self.model.parameters()])

        self.teacher_model.zero_grad()
        with switch_model_status(self.teacher_model, training=False):
            meta_pred = self.teacher_model(labeled_image).softmax(1)
        meta_oh_target = class2one_hot(labeled_target.squeeze(1), C=self.num_classes).float()

        meta_loss = self._meta_criterion(meta_pred, meta_oh_target)
        meta_grad = torch.autograd.grad(meta_loss, tuple(self.teacher_model.parameters()), only_inputs=True)

        self.meters["teacher_loss"].add(meta_loss.item())

        self.model.zero_grad()
        self.teacher_model.load_state_dict(self._teacher_ckpt)
        self.model.load_state_dict(self._student_ckpt)
        self.optimizer.load_state_dict(self._optimizer_ckpt)

        # with autocast(enabled=False):
        # update the weighted gradient.
        with torch.no_grad():
            for i, (meta_g, student_g, p) in enumerate(zip(meta_grad, student_grad, self.model.parameters())):
                p.grad.data.copy_(student_g.data.detach() + self._meta_weight * meta_g.data.detach())

        self.scaler.step(self.optimizer)
        self._updater(ema_model=self.teacher_model, student_model=self.model)
        self.optimizer.zero_grad()
