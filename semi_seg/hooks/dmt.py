from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from typing import Sequence, List

import torch
from loguru import logger
from torch import nn, Tensor
from torch.cuda.amp import GradScaler

from contrastyou.arch import UNet
from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.losses.dice_loss import DiceLoss
from contrastyou.losses.kl import KL_div
from contrastyou.meters import AverageValueMeter, MeterInterface
from contrastyou.types import CriterionType
from contrastyou.utils import class2one_hot, class_name
from contrastyou.hooks import meter_focus
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
        self._updater = EMAUpdater(alpha=alpha, weight_decay=weight_decay, update_bn=True)
        self._model = model
        self._teacher_model = deepcopy(model)
        self._meta_weight = meta_weight
        logger.opt(depth=0).trace("meta_weight: {}".format(meta_weight))
        self._method_name = method_name
        logger.info(f"{class_name(self)} method name: {method_name}")
        self._teacher_optimizer = None
        if method_name in ("method1", "method3", "method4"):
            self._teacher_optimizer = torch.optim.Adam(self._teacher_model.parameters(), lr=meta_weight,
                                                       weight_decay=1e-5)
            self._initialize_teacher_gradient(self._teacher_model, self._teacher_optimizer)

    @staticmethod
    def _initialize_teacher_gradient(teacher_model, teacher_optimizer):
        teacher_model(torch.randn(1, 1, 224, 224, device=next(teacher_model.parameters()).device)).mean().backward()
        teacher_optimizer.zero_grad()

    def __call__(self):
        if self._method_name == "method1":
            return DifferentiableMeanTeacherEpocherHook1(
                name=self._hook_name, model=self._model, weight=self._weight, student_criterion=self._criterion,
                meta_criterion=self._meta_criterion, teacher_model=self.teacher_model, updater=self._updater,
                meta_weight=self._meta_weight, teacher_optimizer=self._teacher_optimizer)
        if self._method_name == "method2":
            return DifferentiableMeanTeacherEpocherHook2(
                name=self._hook_name, model=self._model, weight=self._weight, student_criterion=self._criterion,
                meta_criterion=self._meta_criterion, teacher_model=self.teacher_model, updater=self._updater,
                meta_weight=self._meta_weight, teacher_optimizer=self._teacher_optimizer)
        if self._method_name == "method3":
            return DifferentiableMeanTeacherEpocherHook3(
                name=self._hook_name, model=self._model, weight=self._weight, student_criterion=self._criterion,
                meta_criterion=self._meta_criterion, teacher_model=self.teacher_model, updater=self._updater,
                meta_weight=self._meta_weight, teacher_optimizer=self._teacher_optimizer)
        if self._method_name == "method4":
            return DifferentiableMeanTeacherEpocherHook4(
                name=self._hook_name, model=self._model, weight=self._weight, student_criterion=self._criterion,
                meta_criterion=self._meta_criterion, teacher_model=self.teacher_model, updater=self._updater,
                meta_weight=self._meta_weight, teacher_optimizer=self._teacher_optimizer)
        if self._method_name == "mt":
            return MTEpocherHook(
                name=self._hook_name, model=self._model, weight=self._weight, student_criterion=self._criterion,
                meta_criterion=self._meta_criterion, teacher_model=self.teacher_model, updater=self._updater,
                meta_weight=self._meta_weight, teacher_optimizer=self._teacher_optimizer)
        raise NotImplemented(self._method_name)

    @property
    def teacher_model(self):
        return self._teacher_model

    @property
    def learnable_modules(self) -> List[nn.Module]:
        return []

    @property
    def num_classes(self):
        return self._trainer.num_classes


class _BaseDMTEpocherHook(EpocherHook):
    """
    Base MT Epocher
    """

    def __init__(self, *, name: str, model: nn.Module, teacher_model: nn.Module, student_criterion: CriterionType,
                 meta_criterion: CriterionType, weight: float, meta_weight: float, updater: EMAUpdater,
                 **kwargs) -> None:
        super().__init__(name=name)
        self._model = model
        self._teacher_model = teacher_model

        self._weight = weight
        self._meta_weight: float = meta_weight

        self._criterion = L2LossChecker(student_criterion)
        self._meta_criterion = L2LossChecker(meta_criterion)

        self._updater = updater
        self._teacher_model.train()
        self._model.train()

    def mt_update(self, *, teacher_model, criterion, unlabeled_tf_logits, unlabeled_image, affine_transformer):
        logger.trace("normal mt update.")
        # taken from mean teacher
        student_unlabeled_tf_prob = unlabeled_tf_logits.softmax(1)
        with torch.no_grad():
            teacher_unlabeled_prob = teacher_model(unlabeled_image).softmax(1)
            teacher_unlabeled_prob_tf = affine_transformer(teacher_unlabeled_prob)
        loss = criterion(teacher_unlabeled_prob_tf, student_unlabeled_tf_prob)
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

    @property
    def device(self):
        return self.epocher.device


class MTEpocherHook(_BaseDMTEpocherHook):

    def configure_meters(self, meters: MeterInterface):
        self.meters.register_meter("loss", AverageValueMeter())

    def _call_implementation(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
                             **kwargs):
        loss = self.mt_update(
            teacher_model=self._teacher_model,
            criterion=self._criterion,
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_image=unlabeled_image,
            affine_transformer=affine_transformer
        )
        self.meters["loss"].add(loss.item())
        return self._weight * loss

    def after_batch_update(self, **kwargs):
        # using previous teacher to update the next teacher.
        self._updater(ema_model=self.teacher_model, student_model=self.model)


'''
class SecondDifferentiableMeanTeacherEpocherHook(_BaseDMTEpocherHook, metaclass=ABCMeta):
    """This is not possible to compute the exact value of the second derivative"""

    @abstractmethod
    def not_implemented(self):
        raise NotImplementedError("Not sure of how to implement this baseline.")

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
'''


class DifferentiableMeanTeacherEpocherHook1(_BaseDMTEpocherHook):
    """this implements the update rule 1 of christian's proposal"""

    def __init__(self, *, name: str, model: nn.Module, teacher_model: nn.Module, student_criterion: CriterionType,
                 meta_criterion: CriterionType, weight: float, meta_weight: float, updater: EMAUpdater,
                 teacher_optimizer, **kwargs) -> None:
        super().__init__(name=name, model=model, teacher_model=teacher_model, student_criterion=student_criterion,
                         meta_criterion=meta_criterion, weight=weight, meta_weight=meta_weight, updater=updater,
                         **kwargs)
        self._teacher_optimizer = teacher_optimizer

    def configure_meters(self, meters: MeterInterface):
        self.meters.register_meter("consistency_loss", AverageValueMeter())
        self.meters.register_meter("teacher_loss", AverageValueMeter())

    def _call_implementation(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
                             labeled_image, labeled_target, **kwargs):
        loss = self.mt_update(
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_image=unlabeled_image,
            affine_transformer=affine_transformer,
            teacher_model=self._teacher_model,
            criterion=self._criterion
        )
        self.meters["consistency_loss"].add(loss.item())
        self._teacher_t_ckpt = deepcopy(OrderedDict(self._teacher_model.state_dict()))  # updated bn for teacher
        return self._weight * loss

    @meter_focus
    def after_batch_update(self, labeled_image: Tensor, labeled_target: Tensor, **kwargs):
        self._teacher_optimizer.zero_grad()
        self._updater(ema_model=self._teacher_model, student_model=self.model)
        # teacher at t+1
        labeled_prediction_by_teacher = self._teacher_model(labeled_image).softmax(1)
        onehot_target = class2one_hot(labeled_target.squeeze(1), C=self.num_classes).float()
        meta_loss = self._meta_criterion(labeled_prediction_by_teacher, onehot_target)
        meta_grad = torch.autograd.grad(meta_loss, tuple(self._teacher_model.parameters()), only_inputs=True)
        # gradient for teacher at t+1
        self._teacher_model.load_state_dict(self._teacher_t_ckpt)
        # now teacher at t+1
        with torch.no_grad():
            for i, (meta_g, teacher_p) in enumerate(zip(meta_grad, self.teacher_model.parameters())):
                teacher_p.grad.data.copy_(meta_g.data.detach())
        self._teacher_optimizer.step()
        self.meters["teacher_loss"].add(meta_loss.item())


class DifferentiableMeanTeacherEpocherHook2(_BaseDMTEpocherHook):
    """this implements the update rule 2 of christian's proposal"""

    def configure_meters(self, meters: MeterInterface):
        self.meters.register_meter("consistency_loss", AverageValueMeter())
        self.meters.register_meter("teacher_loss", AverageValueMeter())

    def _call_implementation(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
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


class DifferentiableMeanTeacherEpocherHook3(_BaseDMTEpocherHook):
    """this implements the update rule 3 of christian's proposal"""

    def __init__(self, *, name: str, model: nn.Module, teacher_model: nn.Module, student_criterion: CriterionType,
                 meta_criterion: CriterionType, weight: float, meta_weight: float, updater: EMAUpdater,
                 teacher_optimizer, **kwargs) -> None:
        super().__init__(name=name, model=model, teacher_model=teacher_model, student_criterion=student_criterion,
                         meta_criterion=meta_criterion, weight=weight, meta_weight=meta_weight, updater=updater,
                         **kwargs)
        self._teacher_optimizer = teacher_optimizer

    def configure_meters(self, meters: MeterInterface):
        self.meters.register_meter("consistency_loss", AverageValueMeter())
        self.meters.register_meter("teacher_loss", AverageValueMeter())

    def _call_implementation(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
                             labeled_image, labeled_target, **kwargs):
        loss = self.mt_update(
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_image=unlabeled_image,
            affine_transformer=affine_transformer
        )
        self.meters["consistency_loss"].add(loss.item())
        return self._weight * loss

    def after_batch_update(self, labeled_image: Tensor, labeled_target: Tensor, **kwargs):
        self._teacher_optimizer.zero_grad()
        self._updater(ema_model=self._teacher_model, student_model=self.model)
        # teacher at t+1
        labeled_prediction_by_teacher = self._teacher_model(labeled_image).softmax(1)
        onehot_target = class2one_hot(labeled_target.squeeze(1), C=self.num_classes).float()
        meta_loss = self._meta_criterion(labeled_prediction_by_teacher, onehot_target)
        meta_loss.backward()
        self._teacher_optimizer.step()


class DifferentiableMeanTeacherEpocherHook4(_BaseDMTEpocherHook):
    """this implements the update rule 2 of christian's proposal with an meta optimizer"""

    def __init__(self, *, name: str, model: nn.Module, teacher_model: nn.Module, student_criterion: CriterionType,
                 meta_criterion: CriterionType, weight: float, meta_weight: float, updater: EMAUpdater,
                 teacher_optimizer, **kwargs) -> None:
        super().__init__(name=name, model=model, teacher_model=teacher_model, student_criterion=student_criterion,
                         meta_criterion=meta_criterion, weight=weight, meta_weight=meta_weight, updater=updater,
                         **kwargs)
        self._teacher_optimizer = teacher_optimizer

    @meter_focus
    def configure_meters(self, meters: MeterInterface):
        self.meters.register_meter("consistency_loss", AverageValueMeter())
        self.meters.register_meter("teacher_loss", AverageValueMeter())

    @meter_focus
    def before_regularization(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
                              labeled_image, labeled_target, **kwargs):
        teacher_model = self.teacher_model
        self._teacher_optimizer.zero_grad()
        _ = teacher_model(unlabeled_image).softmax(1)  # to update the bn and save at t
        self._teacher_ckpt_t = deepcopy(OrderedDict(teacher_model.state_dict()))
        with switch_model_status(teacher_model, training=False):
            teacher_labeled_prob = teacher_model(labeled_image).softmax(1)
        one_hot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes).float()

        teacher_loss = self._meta_criterion(teacher_labeled_prob, one_hot_target)
        teacher_loss.backward()
        self._teacher_optimizer.step()
        # teacher at t+1
        self.meters["teacher_loss"].add(teacher_loss.item())

    def _call_implementation(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
                             labeled_image, labeled_target, **kwargs):
        # a temporal teacher at t + 1 from before_regularization
        loss = self.mt_update(teacher_model=self._teacher_model, criterion=self._criterion,
                              unlabeled_image=unlabeled_image, unlabeled_tf_logits=unlabeled_tf_logits,
                              affine_transformer=affine_transformer)
        self.meters["consistency_loss"].add(loss.item())

        return self._weight * loss

    def after_batch_update(self, **kwargs):
        self._teacher_model.load_state_dict(self._teacher_ckpt_t)
        self._updater(ema_model=self._teacher_model, student_model=self.model)


#
# class _DifferentiableMeanTeacherEpocherHook2(DifferentiableMeanTeacherEpocherHook):
#
#     @meter_focus
#     def __call__(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
#                  labeled_image, labeled_target, **kwargs):
#         # student t checkpoint
#
#         loss = self.mt_update(unlabeled_tf_logits=unlabeled_tf_logits, unlabeled_image=unlabeled_image,
#                               affine_transformer=affine_transformer)
#         self.meters["consistency_loss"].add(loss.item())
#
#         self._student_ckpt = deepcopy(self.model.state_dict())
#         self._teacher_ckpt = deepcopy(self.teacher_model.state_dict())
#         self._optimizer_ckpt = deepcopy(self.optimizer.state_dict())
#
#         return self._weight * loss
#
#     @meter_focus
#     def after_batch_update(self, labeled_image, labeled_target, seed, **kwargs):
#         # getting a new teacher
#         self._updater(ema_model=self.teacher_model, student_model=self.model)
#         # backup student gradient
#         student_grad = tuple([x.grad.clone() for x in self.model.parameters()])
#
#         self.teacher_model.zero_grad()
#         with switch_model_status(self.teacher_model, training=False):
#             meta_pred = self.teacher_model(labeled_image).softmax(1)
#         meta_oh_target = class2one_hot(labeled_target.squeeze(1), C=self.num_classes).float()
#
#         meta_loss = self._meta_criterion(meta_pred, meta_oh_target)
#         meta_grad = torch.autograd.grad(meta_loss, tuple(self.teacher_model.parameters()), only_inputs=True)
#
#         self.meters["teacher_loss"].add(meta_loss.item())
#
#         self.model.zero_grad()
#         self.teacher_model.load_state_dict(self._teacher_ckpt)
#         self.model.load_state_dict(self._student_ckpt)
#         self.optimizer.load_state_dict(self._optimizer_ckpt)
#
#         # with autocast(enabled=False):
#         # update the weighted gradient.
#         with torch.no_grad():
#             for i, (meta_g, student_g, p) in enumerate(zip(meta_grad, student_grad, self.model.parameters())):
#                 p.grad.data.copy_(student_g.data.detach() + self._meta_weight * meta_g.data.detach())
#
#         self.scaler.step(self.optimizer)
#         self._updater(ema_model=self.teacher_model, student_model=self.model)
#         self.optimizer.zero_grad()
#
#
# class DifferentiableMeanTeacherEpocherHookBackUp(_BaseDMTEpocherHook):
#     """this implements the update rule 1 of christian's proposal"""
#
#     @meter_focus
#     def configure_meters(self, meters: MeterInterface):
#         self.meters.register_meter("consistency_loss", AverageValueMeter())
#         self.meters.register_meter("teacher_loss", AverageValueMeter())
#
#     @meter_focus
#     def __call__(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
#                  labeled_image, labeled_target, **kwargs):
#         teacher_model = self.teacher_model
#
#         with switch_model_status(teacher_model, training=False):
#             teacher_labeled_prob = teacher_model(labeled_image).softmax(1)
#         one_hot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes).float()
#
#         teacher_loss = self._meta_criterion(teacher_labeled_prob, one_hot_target)
#         _grad_teacher = torch.autograd.grad(teacher_loss, tuple(teacher_model.parameters()), only_inputs=True)
#
#         with manually_forward_with_grad(teacher_model, _grad_teacher, lambda_=self._meta_weight):
#             # manually get the model t+1 and reset to t when exiting
#             student_unlabeled_tf_prob = unlabeled_tf_logits.softmax(1)
#             with torch.no_grad():
#                 teacher_unlabeled_prob = self._teacher_model(unlabeled_image).softmax(1)
#             teacher_unlabeled_prob_tf = affine_transformer(teacher_unlabeled_prob)
#         loss = self._criterion(teacher_unlabeled_prob_tf, student_unlabeled_tf_prob)
#         self.meters["consistency_loss"].add(loss.item())
#         self.meters["teacher_loss"].add(teacher_loss.item())
#
#         return self._weight * loss
#
#     def after_batch_update(self, **kwargs):
#         self._updater(ema_model=self._teacher_model, student_model=self.model)


if __name__ == '__main__':
    model = UNet()
    trainer_hook = DifferentiableMeanTeacherTrainerHook(name="123", model=model, weight=0.1, meta_criterion="ce",
                                                        method_name="method1")
    print(list(nn.ModuleList([trainer_hook]).parameters()))
