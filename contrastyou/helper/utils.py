# dictionary helper functions
import collections
import functools
import warnings
from contextlib import contextmanager
from typing import Union, Dict, Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter  # noqa

__variable_dict = {}


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def toDataLoaderIterator(loader_or_iter: Union[DataLoader, _BaseDataLoaderIter]):
    if not isinstance(loader_or_iter, (_BaseDataLoaderIter, DataLoader)):
        raise TypeError(f"{loader_or_iter} should an instance of DataLoader or _BaseDataLoaderIter, "
                        f"given {loader_or_iter.__class__.__name__}.")
    return loader_or_iter if isinstance(loader_or_iter, _BaseDataLoaderIter) else iter(loader_or_iter)


def get_dataset(dataloader):
    if isinstance(dataloader, _BaseDataLoaderIter):
        return dataloader._dataset
    elif isinstance(dataloader, DataLoader):
        return dataloader.dataset
    else:
        raise NotImplementedError(type(dataloader))


class ChainDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets

    def __len__(self):
        return sum(map(len, self._datasets))

    def __getitem__(self, index: int):
        for i in range(len(self._datasets)):
            try:
                return self._datasets[i][index]
            except IndexError:
                index = index - len(self._datasets[i])


# make a flatten dictionary to be printablely nice.
def nice_dict(input_dict: Dict[str, Union[int, float]]) -> str:
    """
    this function is to return a nice string to dictionary displace propose.
    :param input_dict: dictionary
    :return: string
    """
    assert isinstance(
        input_dict, dict
    ), f"{input_dict} should be a dict, given {type(input_dict)}."
    is_flat_dict = True
    for k, v in input_dict.items():
        if isinstance(v, dict):
            is_flat_dict = False
            break
    flat_dict = input_dict if is_flat_dict else flatten_dict(input_dict, sep="")
    string_list = [f"{k}:{v:.3f}" for k, v in flat_dict.items()]
    return ", ".join(string_list)


def average_iter(a_list):
    return sum(a_list) / float(len(a_list))


def multiply_iter(iter_a, iter_b):
    return [x * y for x, y in zip(iter_a, iter_b)]


def weighted_average_iter(a_list, weight_list):
    sum_weight = sum(weight_list) + 1e-16
    return sum(multiply_iter(a_list, weight_list)) / sum_weight


def pairwise_distances(x, y=None, recall_func=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
           recall function is a function to manipulate the distance.
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    if recall_func:
        return recall_func(dist)
    return dist


@contextmanager
def plt_interactive():
    plt.ion()
    yield
    plt.ioff()


def extract_model_state_dict(trainer_checkpoint_path: str):
    trainer_state = torch.load(trainer_checkpoint_path, map_location="cpu")

    return trainer_state["_model"]


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def register_variable(*, name: str, object_: Any):
    __variable_dict[name] = object_


def get_variable(*, name: str):
    return __variable_dict[name]
