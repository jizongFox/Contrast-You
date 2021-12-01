import torch
from torch.nn import MSELoss

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.meters import AverageValueMeter, MeterInterface
from contrastyou.utils import class2one_hot


class PseudoLabelTrainerHook(TrainerHook):

    def __init__(self, name: str, weight: float):
        super().__init__(hook_name=name)
        self._weight = weight
        self._criterion = MSELoss()

    def __call__(self):
        return _PLEpocherHook(name=self._hook_name, weight=self._weight, criterion=self._criterion)


class _PLEpocherHook(EpocherHook):
    def __init__(self, name: str, weight: float, criterion) -> None:
        super().__init__(name=name)
        self._weight = weight
        self._criterion = criterion

    def configure_meters_given_epocher(self, meters: MeterInterface):
        self.meters.register_meter("loss", AverageValueMeter())

    def _call_implementation(self, *, unlabeled_tf_logits, unlabeled_logits_tf, seed, affine_transformer, **kwargs):
        unlabeled_prob_tf = unlabeled_logits_tf.softmax(1)
        C = unlabeled_prob_tf.shape[1]
        with torch.no_grad():
            pseudo_label = unlabeled_prob_tf.max(1)[1]
            oh_label = class2one_hot(pseudo_label, C).float()

        loss = self._criterion(unlabeled_prob_tf, oh_label)
        self.meters["loss"].add(loss.item())
        return self._weight * loss
