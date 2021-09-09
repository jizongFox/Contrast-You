# this hook makes the prototype weights to be orthogonal
from torch import Tensor
from torch.nn import functional as F

from contrastyou.hooks import TrainerHook, EpocherHook
from contrastyou.meters import MeterInterface, AverageValueMeter
from semi_seg.hooks import meter_focus


def pairwise_matrix(vec1: Tensor, vec2: Tensor):
    assert vec1.shape == vec2.shape
    assert vec1.dim() == 2, vec1.shape
    return vec1 @ vec2.t()


def normalize(vec: Tensor, dim: int = 1):
    return F.normalize(vec, p=2, dim=dim)


class OrthogonalTrainerHook(TrainerHook):

    def __init__(self, *, hook_name: str, prototypes: Tensor, weight: float = 0.0):
        super().__init__(hook_name=hook_name)
        self._weight = weight
        self._prototype_weights = prototypes

    def __call__(self):
        return _OrthogonalEpocherHook(name=self._hook_name, prototypes=self._prototype_weights, weight=self._weight)


class _OrthogonalEpocherHook(EpocherHook):

    def __init__(self, *, name: str, prototypes: Tensor, weight: float) -> None:
        super().__init__(name=name)
        self._prototypes = prototypes
        self._weight = weight

    @meter_focus
    def configure_meters(self, meters: MeterInterface):
        meters.register_meter("loss", AverageValueMeter())
        return meters

    @meter_focus
    def __call__(self, **kwargs):
        normalized_prototypes = normalize(self._prototypes)
        matrix = pairwise_matrix(normalized_prototypes.squeeze(), normalized_prototypes.squeeze())
        loss = matrix.mean()
        self.meters["loss"].add(loss.item())
        return loss * self._weight
