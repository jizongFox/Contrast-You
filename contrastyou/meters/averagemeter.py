from collections import defaultdict
from typing import List

import numpy as np

from .metric import Metric


class AverageValueMeter(Metric):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()

    def _add(self, value, n=1):
        self.sum += value * n
        self.n += n

    def reset(self):
        self.sum = 0
        self.n = 0

    def _summary(self) -> float:
        # this function returns a dict and tends to aggregate the historical results.
        if self.n == 0:
            return np.nan
        return float(self.sum / self.n)


class AverageValueDictionaryMeter(Metric):
    def __init__(self) -> None:
        super().__init__()
        self._meter_dicts = defaultdict(AverageValueMeter)

    def reset(self):
        for k, v in self._meter_dicts.items():
            v.reset()

    def _add(self, **kwargs):
        for k, v in kwargs.items():
            self._meter_dicts[k].add(v)

    def _summary(self):
        return {k: v.summary() for k, v in self._meter_dicts.items()}


class AverageValueListMeter(AverageValueDictionaryMeter):
    def _add(self, list_value: List[float], **kwargs):
        for i, v in enumerate(list_value):
            self._meter_dicts[str(i)].add(v)
