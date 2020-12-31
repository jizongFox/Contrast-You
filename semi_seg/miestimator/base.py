from abc import ABCMeta, abstractmethod

from torch import nn


class _SingleEstimator(nn.Module):
    pass


class _EstimatorList(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        self._estimator_list = nn.ModuleDict()

    @abstractmethod
    def add(self, name: str, **params):
        ...

    @abstractmethod
    def remove(self, name: str, **params):
        ...
