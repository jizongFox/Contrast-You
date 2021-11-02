# this hook makes the prototype weights to be orthogonal
import torch
from loguru import logger
from torch import Tensor
from torch.nn import functional as F

from contrastyou.hooks import TrainerHook, EpocherHook
from contrastyou.meters import MeterInterface, AverageValueMeter
from contrastyou.utils import class_name


def pairwise_matrix(vec1: Tensor, vec2: Tensor):
    assert vec1.shape == vec2.shape
    assert vec1.dim() == 2
    return vec1 @ vec2.t()


def normalize(vec: Tensor, dim: int = 1):
    return F.normalize(vec, p=2, dim=dim)


class OrthogonalTrainerHook(TrainerHook):

    def __init__(self, *, hook_name: str, prototypes: Tensor, weight: float = 0.0):
        super().__init__(hook_name=hook_name)
        self._weight = weight
        self._prototype_weights = prototypes
        logger.debug(f"Created {class_name(self)} with name: {self._hook_name}, weight: {self._weight}.")

    def __call__(self):
        return _OrthogonalEpocherHook(name=self._hook_name, prototypes=self._prototype_weights, weight=self._weight)


class _OrthogonalEpocherHook(EpocherHook):

    def __init__(self, *, name: str, prototypes: Tensor, weight: float) -> None:
        super().__init__(name=name)
        self._prototypes = prototypes
        self._weight = weight

    def configure_meters_given_epocher(self, meters: MeterInterface):
        meters.register_meter("loss", AverageValueMeter())
        return meters

    def _call_implementation(self, **kwargs):
        normalized_prototypes = normalize(self._prototypes)
        matrix = pairwise_matrix(normalized_prototypes.squeeze(), normalized_prototypes.squeeze())
        loss = (matrix - torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)).pow(2).mean()
        self.meters["loss"].add(loss.item())
        return loss * self._weight
