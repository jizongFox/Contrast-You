import math

import torch
from contrastyou.helper import average_iter
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats as disable_bn
from deepclustering2.loss import Entropy
from deepclustering2.meters2 import MeterInterface, AverageValueMeter
from deepclustering2.models import ema_updater as EMA_Updater
from deepclustering2.schedulers.customized_scheduler import RampScheduler
from deepclustering2.type import T_loss
from torch import Tensor, nn

from .comparable import MeanTeacherEpocher


class UCMeanTeacherEpocher(MeanTeacherEpocher):

    def init(self, *, reg_weight: float, teacher_model: nn.Module, reg_criterion: T_loss, ema_updater: EMA_Updater,
             threshold: RampScheduler = None, **kwargs):
        super().init(reg_weight=reg_weight, teacher_model=teacher_model, reg_criterion=reg_criterion,
                     ema_updater=ema_updater, **kwargs)
        assert isinstance(threshold, RampScheduler), threshold
        self._threshold: RampScheduler = threshold
        self._entropy_loss = Entropy(reduction="none")

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(UCMeanTeacherEpocher, self)._configure_meters(meters)
        meters.register_meter("uc_weight", AverageValueMeter())
        meters.register_meter("uc_ratio", AverageValueMeter())
        return meters

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int,
                       unlabeled_image: Tensor, unlabeled_image_tf: Tensor, *args, **kwargs):
        @torch.no_grad()
        def get_teacher_pred_with_tf(uimage, noise=None):
            if noise is not None:
                uimage += noise
            teacher_unlabeled_logit = self._teacher_model(uimage)
            with FixRandomSeed(seed):
                teacher_unlabeled_logit_tf = torch.stack(
                    [self._affine_transformer(x) for x in teacher_unlabeled_logit], dim=0)
            return teacher_unlabeled_logit_tf

        # compare teacher_unlabeled_logit_tf and student unlabeled_tf_logits
        self._reg_criterion.reduction = "none"  # here the self._reg_criterion should be nn.MSELoss or KLDiv()
        teacher_unlabeled_logit_tf = get_teacher_pred_with_tf(unlabeled_image)
        reg_loss = self._reg_criterion(unlabeled_tf_logits.softmax(1), teacher_unlabeled_logit_tf.softmax(1).detach())

        # uncertainty:
        with disable_bn(self._teacher_model):
            uncertainty_predictions = [
                get_teacher_pred_with_tf(unlabeled_image, 0.05 * torch.randn_like(unlabeled_image)) for _ in range(8)
            ]

        average_prediction = average_iter(uncertainty_predictions)

        entropy = self._entropy_loss(average_prediction.softmax(1)) / math.log(average_prediction.shape[1])
        th = self._threshold.value
        mask = (entropy <= th).float()

        self.meters["uc_weight"].add(th)
        self.meters["uc_ratio"].add(mask.mean().item())

        # update teacher model here.
        self._ema_updater(self._teacher_model, self._model)
        return (reg_loss.mean(1) * mask).mean()
