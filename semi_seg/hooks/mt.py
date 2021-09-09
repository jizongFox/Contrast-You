from copy import deepcopy
from typing import List, Tuple

import torch
from loguru import logger
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.meters import AverageValueMeter, MeterInterface
from contrastyou.utils import simplex, class2one_hot
from semi_seg.hooks import meter_focus


def pair_iterator(model_list: List[nn.Module]) -> Tuple[nn.Module, nn.Module]:
    assert len(model_list) >= 2, len(model_list)
    current_model = model_list.pop(0)
    next_model = model_list.pop(0)
    while True:
        try:
            yield current_model, next_model
            current_model = next_model
            next_model = model_list.pop(0)
        except IndexError:
            break


class L2LossChecker:
    """this checker is to check if the input of the criterion to be simplex"""

    def __init__(self, criterion) -> None:
        super().__init__()
        self.criterion = criterion

    def __call__(self, input1, input2):
        assert simplex(input1) and simplex(input2)
        loss = self.criterion(input1, input2)
        if torch.isnan(loss):
            raise RuntimeError(loss)
        return loss


class EMAUpdater:
    def __init__(
            self, alpha=0.999, justify_alpha=True, weight_decay=1e-5, update_bn=False
    ) -> None:
        self._alpha = alpha
        self._weight_decay = weight_decay
        self._update_bn = update_bn
        self._justify_alpha = justify_alpha
        self.__global_step = 0

    @torch.no_grad()
    def __call__(self, ema_model: nn.Module, student_model: nn.Module):
        alpha = self._alpha
        if self._justify_alpha:
            alpha = min(1 - 1 / (self.__global_step + 1), self._alpha)

        for ema_param, s_param in zip(
                ema_model.parameters(), student_model.parameters()
        ):
            ema_param.data.mul_(alpha).add_(1 - alpha, s_param.data)
            if self._weight_decay > 0:
                ema_param.data.mul_(1 - self._weight_decay)

        if self._update_bn:
            # running mean and vars for bn
            for (name, ema_buffer), (_, s_buffer) in zip(
                    ema_model.named_buffers(), student_model.named_buffers(),
            ):
                if "running_mean" in name or "running_var" in name:
                    ema_buffer.data.mul_(alpha).add_(1 - alpha, s_buffer.data)
                    if self._weight_decay > 0:
                        ema_buffer.data.mul_(1 - self._weight_decay)

        self.__global_step += 1


class MeanTeacherTrainerHook(TrainerHook):

    def __init__(self, name: str, model: nn.Module, weight: float, alpha: float = 0.999, weight_decay: float = 1e-5,
                 update_bn=False, num_teachers: int = 1, hard_clip=False):
        """
        adding parameters: num_teachers to host multiple teacher model
        The first model is going to update the bn or not but the following models must update bn by force
        """
        super().__init__(hook_name=name)
        self._weight = weight
        self._criterion = nn.MSELoss()
        if update_bn:
            logger.info("set all bn to be eval model")
        self._updater = EMAUpdater(alpha=alpha, weight_decay=weight_decay, update_bn=update_bn)
        self._teacher_model = deepcopy(model)

        self._num_teachers = num_teachers
        self._extra_teachers = nn.ModuleList()
        # update extra teacher by force.
        self._extra_teacher_updater = EMAUpdater(alpha=alpha, weight_decay=weight_decay, update_bn=True)
        self._hard_clip = hard_clip

        if num_teachers > 1:
            logger.debug(f"Initializing {num_teachers} extra teachers")
            self._extra_teachers.extend([deepcopy(model) for _ in range(num_teachers - 1)])

        for p in self._teacher_model.parameters():
            p.detach_()
        for model in self._extra_teachers:
            for p in model.parameters():
                p.detach_()

    def __call__(self):
        return _MeanTeacherEpocherHook(name=self._hook_name, weight=self._weight, criterion=self._criterion,
                                       teacher_model=self._teacher_model, updater=self._updater,
                                       extra_teachers=self._extra_teachers, extra_updater=self._extra_teacher_updater,
                                       hard_clip=self._hard_clip)

    @property
    def teacher_model(self):
        return self._teacher_model

    def get_teacher(self, index: int = 0):
        assert index < self._num_teachers
        if index == 0:
            return self._teacher_model
        return self._extra_teachers[index - 1]

    def has_extra_mt(self):
        return self._num_teachers > 1

    @property
    def extra_teachers(self):
        return self._extra_teachers

    @property
    def learnable_modules(self) -> List[nn.Module]:
        return [x for x in self.extra_teachers]


class _MeanTeacherEpocherHook(EpocherHook):
    def __init__(self, name: str, weight: float, criterion, teacher_model, updater, extra_teachers,
                 extra_updater, hard_clip: bool = False) -> None:
        super().__init__(name)
        self._weight = weight
        self._criterion = L2LossChecker(criterion)
        self._teacher_model = teacher_model
        self._updater = updater
        self._hard_clip = hard_clip

        self._teacher_model.train()
        if updater._update_bn:  # noqa
            # if update bn, then, freeze all bn in the network to eval()
            for m in self._teacher_model.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

        extra_teachers.train()
        # set extra teachers to be eval for bn
        for model in extra_teachers:
            for m in model.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

        self._extra_teachers = extra_teachers
        self._extra_updater = extra_updater

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
            if self._hard_clip:
                C = teacher_unlabeled_prob_tf.shape[1]
                teacher_unlabeled_prob_tf = teacher_unlabeled_prob_tf.max(1)[1]
                teacher_unlabeled_prob_tf = class2one_hot(teacher_unlabeled_prob_tf, C).float()
        loss = self._criterion(teacher_unlabeled_prob_tf, student_unlabeled_tf_prob)
        self.meters["loss"].add(loss.item())
        return self._weight * loss

    def after_batch_update(self, **kwargs):
        # using previous teacher to update the next teacher.
        self._updater(ema_model=self._teacher_model, student_model=self.model)

        if len(self._extra_teachers) > 0:
            teachers_to_update = [self._teacher_model, *self._extra_teachers]
            teacher_iter = pair_iterator(teachers_to_update)
            for previous_t, next_t in teacher_iter:
                self._extra_updater(ema_model=next_t, student_model=previous_t)

    @property
    def model(self):
        return self.epocher._model  # noqa

    @property
    def teacher_model(self):
        return self._teacher_model
