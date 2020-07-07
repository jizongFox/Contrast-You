import functools
from abc import ABCMeta
from collections import defaultdict
from typing import DefaultDict, Callable, List, Dict

import pandas as pd

from deepclustering2.utils import path2Path
from ._historical_container import HistoricalContainer
from ._utils import rename_df_columns

__all__ = ["Storage"]


class _IOMixin:
    _storage: DefaultDict[str, HistoricalContainer]
    summary: Callable[[], pd.DataFrame]

    def state_dict(self):
        return self._storage

    def load_state_dict(self, state_dict):
        self._storage = state_dict

    def to_csv(self, path, name="storage.csv"):
        path = path2Path(path)
        assert path.is_dir(), path
        path.mkdir(exist_ok=True, parents=True)
        self.summary().to_csv(path / name)


class Storage(_IOMixin, metaclass=ABCMeta):

    def __init__(self) -> None:
        super().__init__()
        self._storage = defaultdict(HistoricalContainer)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __getitem__(self, item):
        if item not in self._storage:
            raise KeyError(f"{item} not found in {__class__.__name__}")
        return self._storage[item]

    def put(self, name: str, value: Dict[str, float], epoch=None, prefix="", postfix=""):
        self._storage[prefix + name + postfix].add(value, epoch)

    def put_all(self, report_dict, epoch=None):
        for k, v in report_dict.items():
            self.put(k, v, epoch)

    def get(self, name, epoch=None):
        assert name in self._storage, name
        if epoch is None:
            return self._storage[name]
        return self._storage[name][epoch]

    def summary(self) -> pd.DataFrame:
        """
        summary on the list of sub summarys, merging them together.
        :return:
        """
        result_dict = {}
        for k, v in self._storage.items():
            result_dict[k]=v.record_dict
        # flatten the dict
        from deepclustering.utils import flatten_dict
        flatten_result = flatten_dict(result_dict)
        return pd.DataFrame(flatten_result)

    @property
    def meter_names(self, sorted=False) -> List[str]:
        if sorted:
            return sorted(self._storage.keys())
        return list(self._storage.keys())

    @property
    def storage(self):
        return self._storage
