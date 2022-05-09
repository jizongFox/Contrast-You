import atexit
import contextlib
import math
import typing as t
from functools import wraps

from loguru import logger

from tqdm import tqdm as _tqdm
from ..utils.printable import item2str

if t.TYPE_CHECKING:
    from contrastyou.epochers.base import EpocherBase


def create_meter_display(group_dict: t.Dict, ignore_start_with="_"):
    def prune_dict(dictionary: dict, ignore="_"):
        for k, v in dictionary.copy().items():
            if isinstance(v, dict):
                prune_dict(v, ignore)
            else:
                if k.startswith(ignore):
                    del dictionary[k]

    def prune_nan(dictionary: dict, father_dictionary: dict = None):

        for k, v in dictionary.copy().items():
            if isinstance(v, dict):
                prune_nan(v, dictionary)
            else:
                if math.isnan(v):
                    del dictionary[k]
                    if father_dictionary is not None:
                        if len(father_dictionary) == 1:
                            del father_dictionary

    prune_dict(group_dict, ignore_start_with)

    display = str(item2str(group_dict))
    return display


class frequency_cache:

    def __init__(self, freq: int = 10) -> None:
        self.freq = freq
        self._n = 0

        self._cache = None
        self.__func__ = None

    def __call__(self, func):
        self.__func__ = func

        @wraps(func)
        def wrapper(*args, force_update, **kwargs):
            self._n += 1
            if force_update is True or self._n == 1:
                self.cache = func(*args, **kwargs)
                return self.cache
            if self._n % self.freq == 0:
                self.cache = func(*args, **kwargs)
                return self.cache
            return self.cache

        return wrapper


class tqdm(_tqdm):

    def __init__(self, iterable=None, desc=None, total=None, leave=False, file=None, ncols=None, mininterval=0.1,
                 maxinterval=10.0, miniters=None, ascii=None, disable=False, unit='it', unit_scale=False,
                 dynamic_ncols=True, smoothing=0.3,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [" "{rate_fmt}{postfix}]", initial=0, position=None,
                 postfix=None, unit_divisor=1000, write_bytes=None, gui=False, **kwargs):
        super().__init__(iterable, desc, total, leave, file, ncols, mininterval, maxinterval, miniters, ascii, disable,
                         unit, unit_scale, dynamic_ncols, smoothing, bar_format, initial, position, postfix,
                         unit_divisor, write_bytes, gui, **kwargs)

        self._cur_iter = 0
        atexit.register(self.close)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def log_result(self):
        if hasattr(self, "__cached__"):
            logger.opt(depth=3).info(self.desc + "    " + create_meter_display(self.__cached__))

    def __enter__(self):
        return self

    def set_desc_from_epocher(self, epocher: "EpocherBase"):
        des = f"{epocher.__class__.__name__:<15} {epocher._cur_epoch:03d}"
        return self.set_description(desc=des)

    def set_postfix_statics(self, group_dictionary, group_iter_time=None, cache_time=1):
        self._cur_iter += 1
        if not hasattr(self, "__cached__"):
            self.__cached__ = dict(group_dictionary)
            self._group_keys = list(self.__cached__.keys())
        if self._cur_iter % cache_time == 0:
            self.__cached__ = dict(group_dictionary)  # noqa
            self._group_keys = list(self.__cached__.keys())  # noqa
        if group_iter_time is None:
            return self._set_postfix_statics(self.__cached__)
        group_index = self._cur_iter // group_iter_time % len(self._group_keys)
        group_name = self._group_keys[group_index]
        if len(self._group_keys) == 1:
            return self._set_postfix_statics(self.__cached__[group_name])
        return self._set_postfix_statics({group_name: self.__cached__[group_name]})

    def _set_postfix_statics(self, dict2display):
        pretty_str = create_meter_display(dict2display)
        self.set_postfix_str(pretty_str)

    @frequency_cache(freq=10)
    def set_postfix_statics2(self, group_dictionary, force_update: bool = False):
        return self._set_postfix_statics(dict(group_dictionary))

    @contextlib.contextmanager
    def disable_cache(self):
        orig_func = self.set_postfix_statics2
        self.set_postfix_statics2 = self.set_postfix_statics2.__func__
        yield
        self.set_postfix_statics2 = orig_func
