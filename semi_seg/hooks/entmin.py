from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.losses.kl import Entropy
from contrastyou.meters import AverageValueMeter, MeterInterface
from contrastyou.utils import simplex


class EntropyMinTrainerHook(TrainerHook):

    def __init__(self, name: str, weight: float):
        super().__init__(hook_name=name)
        self._weight = weight
        self._criterion = Entropy()

    def __call__(self):
        return _EntropyEpocherHook(name=self._hook_name, weight=self._weight, criterion=self._criterion)


class _EntropyEpocherHook(EpocherHook):
    def __init__(self, name: str, weight: float, criterion) -> None:
        super().__init__(name=name)
        self._weight = weight
        self._criterion = criterion

    def configure_meters_given_epocher(self, meters: MeterInterface):
        self.meters.register_meter("loss", AverageValueMeter())

    def _call_implementation(self, *, unlabeled_tf_logits, unlabeled_logits_tf, seed, affine_transformer, **kwargs):
        assert not simplex(unlabeled_logits_tf)
        unlabeled_prob_tf = unlabeled_logits_tf.softmax(1)
        loss = self._criterion(unlabeled_prob_tf)
        self.meters["loss"].add(loss.item())
        return self._weight * loss
