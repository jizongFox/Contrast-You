import math
import typing as t
from copy import deepcopy

import torch
from loguru import logger
from torch import nn, Tensor
from torch.nn.modules.batchnorm import _BatchNorm  # noqa

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.meters import AverageValueMeter, MeterInterface
from contrastyou.utils import simplex, class2one_hot, fix_all_seed_within_context
from semi_seg.hooks.utils import mixup_data


def pair_iterator(model_list: t.List[nn.Module]) -> t.Iterable[t.Tuple[nn.Module, nn.Module]]:
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


def detach_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.detach_()


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

    def __init__(self, *, name: str, model: nn.Module, weight: float, alpha: float = 0.999, weight_decay: float = 1e-5,
                 update_bn=False, num_teachers: int = 1, hard_clip=False):
        """
        adding parameters: num_teachers to host multiple teacher model
        The first model is going to update the bn or not but the following models must update bn by force
        """
        super().__init__(hook_name=name)
        if num_teachers > 1:
            raise RuntimeError(f"Current version only support one Teacher, given {num_teachers} Teachers.")

        self._weight = weight
        self._criterion = nn.MSELoss(reduction="none")  # change the mse to reduction none
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

        detach_model(self._teacher_model)
        for _model in self._extra_teachers:
            detach_model(_model)

    def __call__(self):
        return _MeanTeacherEpocherHook(
            name=self._hook_name, weight=self._weight, criterion=self._criterion,
            teacher_model=self._teacher_model, updater=self._updater,
            extra_teachers=self._extra_teachers, extra_updater=self._extra_teacher_updater,
            hard_clip=self._hard_clip
        )

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
    def learnable_modules(self) -> t.List[nn.Module]:
        return [x for x in self.extra_teachers]


class _MeanTeacherEpocherHook(EpocherHook):
    def __init__(self, *, name: str, weight: float, criterion, teacher_model, updater, extra_teachers,
                 extra_updater, hard_clip: bool = False) -> None:
        super().__init__(name=name)
        self._weight = weight
        self._criterion = criterion  # l2checker can break the pipeline if padding values are given.
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

    def configure_meters_given_epocher(self, meters: MeterInterface):
        self.meters.register_meter("loss", AverageValueMeter())

    def _call_implementation(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
                             **kwargs):
        C = unlabeled_tf_logits.shape[1]
        student_unlabeled_tf_prob = unlabeled_tf_logits.softmax(1)
        with torch.no_grad():
            teacher_unlabeled_logit = self.teacher_model(unlabeled_image)
            teacher_unlabeled_prob_tf = affine_transformer(teacher_unlabeled_logit, mode="feature").softmax(1)

            if self._hard_clip:
                teacher_unlabeled_prob_tf = teacher_unlabeled_prob_tf.argmax(1)
                teacher_unlabeled_prob_tf = class2one_hot(teacher_unlabeled_prob_tf, C).float()

        loss = self._criterion(teacher_unlabeled_prob_tf, student_unlabeled_tf_prob).mean()
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


class UAMeanTeacherTrainerHook(MeanTeacherTrainerHook):

    def __call__(self):
        return _UAMeanTeacherEpocherHook(
            name=self._hook_name, weight=self._weight, criterion=self._criterion,
            teacher_model=self._teacher_model, updater=self._updater,
            extra_teachers=self._extra_teachers, extra_updater=self._extra_teacher_updater,
            hard_clip=self._hard_clip
        )


class _UAMeanTeacherEpocherHook(_MeanTeacherEpocherHook):

    def configure_meters_given_epocher(self, meters: MeterInterface):
        super().configure_meters_given_epocher(meters)
        self.meters.register_meter("mask", AverageValueMeter())

    def _call_implementation(self, *,
                             unlabeled_tf_logits, unlabeled_image, unlabeled_image_tf=None, seed,
                             affine_transformer, **kwargs):
        assert unlabeled_image_tf is not None
        C = unlabeled_tf_logits.shape[1]
        student_unlabeled_tf_prob = unlabeled_tf_logits.softmax(1)
        with torch.no_grad():
            teacher_unlabeled_prob_tf, entropy_tf = self._aggregate_predictions(
                unlabeled_image=unlabeled_image,
                N=4,
                affine_transformer=affine_transformer,
                seed=seed
            )
            if self._hard_clip:
                teacher_unlabeled_prob_tf = teacher_unlabeled_prob_tf.argmax(1)
                teacher_unlabeled_prob_tf = class2one_hot(teacher_unlabeled_prob_tf, C).float()
        loss = self._criterion(teacher_unlabeled_prob_tf, student_unlabeled_tf_prob).mean(1)
        mask = (entropy_tf < 3 / 4 * math.log(C) + 1 / 4 * math.log(C) * float(self.cur_epoch / self.max_epoch)).float()
        loss = loss * mask
        loss = loss.mean() / (mask.mean().item() + 1e-2)
        self.meters["loss"].add(loss.item())
        self.meters["mask"].add(mask.mean().item())
        return self._weight * loss

    @torch.no_grad()
    @logger.contextualize(enabled=False)
    def _aggregate_predictions(self, *, unlabeled_image: Tensor, N: int = 8, affine_transformer, seed: int) \
            -> t.Tuple[Tensor, Tensor]:
        with self.teacher_model.switch_bn_track(enable=True):  # this is to write bn statistics
            teacher_unlabeled_logit = self.teacher_model(unlabeled_image)

        with self.teacher_model.switch_bn_track(enable=False), \
                fix_all_seed_within_context(seed):
            teacher_unlabeled_logits = [
                self.teacher_model(unlabeled_image + torch.randn_like(unlabeled_image) * 0.05) for _ in range(N)
            ]
        teacher_unlabeled_logit = torch.mean(
            torch.stack([teacher_unlabeled_logit, *teacher_unlabeled_logits], dim=0),
            dim=0
        )
        teacher_unlabeled_prob_tf = affine_transformer(teacher_unlabeled_logit, mode="feature").softmax(1)
        entropy_tf = -(teacher_unlabeled_prob_tf * (teacher_unlabeled_prob_tf + 1e-16).log()).sum(1)
        return teacher_unlabeled_prob_tf, entropy_tf

    @property
    def cur_epoch(self) -> int:
        return self.epocher.cur_epoch

    @property
    def max_epoch(self) -> int:
        return self.epocher.trainer._max_epoch


class ICTMeanTeacherTrainerHook(MeanTeacherTrainerHook):

    def __init__(self, *, name: str, model: nn.Module, weight: float, alpha: float = 0.999, weight_decay: float = 1e-5,
                 update_bn=False, ):
        super().__init__(name=name, model=model, weight=weight, alpha=alpha, weight_decay=weight_decay,
                         update_bn=update_bn, num_teachers=1, hard_clip=False)

    def __call__(self):
        return _ICTMeanTeacherEpocherHook(
            name=self._hook_name, weight=self._weight, criterion=self._criterion,
            teacher_model=self._teacher_model, updater=self._updater,
            extra_teachers=self._extra_teachers, extra_updater=self._extra_teacher_updater,
        )


class _ICTMeanTeacherEpocherHook(_MeanTeacherEpocherHook):

    def __init__(self, *, name: str, weight: float, criterion, teacher_model, updater, extra_teachers,
                 extra_updater) -> None:
        super().__init__(name=name, weight=weight, criterion=criterion, teacher_model=teacher_model, updater=updater,
                         extra_teachers=extra_teachers, extra_updater=extra_updater, hard_clip=False)

    def _call_implementation(
            self, *, unlabeled_tf_logits, unlabeled_image: Tensor, unlabeled_image_tf, seed,  # noqa
            **kwargs):  # noqa

        with torch.no_grad():
            teacher_unlabeled_prob = self.teacher_model(unlabeled_image).softmax(1)
            teacher_unlabeled_tf_prob = self.teacher_model(unlabeled_image_tf).softmax(1)
            with fix_all_seed_within_context(seed):
                mixed_image, mixed_target = mixup_data(
                    x=torch.cat([unlabeled_image, unlabeled_image_tf], dim=0),
                    y=torch.cat([teacher_unlabeled_prob, teacher_unlabeled_tf_prob], dim=0),
                    alpha=0.2, device=unlabeled_image.device
                )
        student_mixed_prob = self.model(mixed_image).softmax(1)

        loss = self._criterion(student_mixed_prob, mixed_target.squeeze()).mean()
        if self.meters:
            self.meters["loss"].add(loss.item())
        return self._weight * loss
