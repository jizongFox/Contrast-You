from abc import ABCMeta, abstractmethod

from torch import nn


class _SingleEstimator(nn.Module):
    pass


class _EstimatorList(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        self._estimator_dictionary = nn.ModuleDict()

    @abstractmethod
    def add(self, name: str, **params):
        ...

    def remove(self, name: str):
        assert name not in self._estimator_dictionary, self._estimator_dictionary.keys()
        self._estimator_dictionary.remove(name)

    def __iter__(self):
        # enable for iteration
        for v in self._estimator_dictionary.values():
            yield v

    def __len__(self):
        # enable len() operation
        return len(self._estimator_dictionary)
