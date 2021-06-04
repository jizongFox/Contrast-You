from copy import deepcopy

import torch
from torch import nn

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.meters import AverageValueMeter, MeterInterface
from semi_seg.hooks import meter_focus


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

    def __init__(self, name: str, weight: float, model: nn.Module):
        super().__init__(name)
        self._weight = weight
        self._criterion = nn.MSELoss()
        self._updater = EMAUpdater()
        self._teacher_model = deepcopy(model)
        for p in self._teacher_model.parameters():
            p.detach_()

    def __call__(self):
        return _MeanTeacherEpocherHook(name=self._hook_name, weight=self._weight, criterion=self._criterion,
                                       teacher_model=self._teacher_model, updater=self._updater)

    @property
    def teacher_model(self):
        return self._teacher_model


class _MeanTeacherEpocherHook(EpocherHook):
    def __init__(self, name: str, weight: float, criterion, teacher_model, updater) -> None:
        super().__init__(name)
        self._weight = weight
        self._criterion = criterion
        self._teacher_model = teacher_model
        self._updater = updater

    @meter_focus
    def configure_meters(self, meters: MeterInterface):
        self.meters.register_meter("loss", AverageValueMeter())

    @meter_focus
    def __call__(self, *, unlabeled_tf_logits, unlabeled_image, seed, affine_transformer,
                 **kwargs):
        student_unlabeled_tf_prob = unlabeled_tf_logits.softmax(1)
        teacher_unlabeled_prob = self._teacher_model(unlabeled_image).softmax(1)
        teacher_unlabeled_prob_tf = affine_transformer(teacher_unlabeled_prob, seed=seed)
        loss = self._criterion(teacher_unlabeled_prob_tf, student_unlabeled_tf_prob)
        self.meters["loss"].add(loss.item())
        self._updater(ema_model=self._teacher_model, student_model=self.epocher._model)  # noqa
        return self._weight * loss
