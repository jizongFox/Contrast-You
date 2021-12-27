from abc import ABCMeta
from collections import OrderedDict
from typing import Dict, Any

import pandas as pd


def OrderedDict2DataFrame(dictionary: Dict[int, Dict]):
    try:
        validated_table = pd.DataFrame(dictionary).T
    except ValueError:
        validated_table = pd.DataFrame(dictionary, index=[""]).T
    return validated_table


class HistoricalContainer(metaclass=ABCMeta):
    """
    Aggregate historical information in a ordered dict.
    """

    def __init__(self) -> None:
        self._record_dict: "OrderedDict" = OrderedDict()
        self._current_epoch = 0

    @property
    def record_dict(self) -> OrderedDict:
        return self._record_dict

    def __getitem__(self, index):
        return self._record_dict[index]

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    def summary(self) -> pd.DataFrame:
        return OrderedDict2DataFrame(self._record_dict)

    def add(self, input_dict, epoch=None) -> None:
        if epoch:
            self._current_epoch = epoch
        self._record_dict[self._current_epoch] = input_dict
        self._current_epoch += 1

    def reset(self) -> None:
        self._record_dict = OrderedDict()
        self._current_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def __repr__(self):
        return str(self.summary())


def rename_df_columns(dataframe: pd.DataFrame, name: str, sep="_"):
    dataframe.columns = list(map(lambda x: name + sep + x, dataframe.columns))
    return dataframe
