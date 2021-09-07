from itertools import product

import numpy as np
import os
import random
import string
import torch
from collections.abc import Iterable
from contrastyou import PROJECT_PATH, on_cc
from pathlib import Path
# from semi_seg import data2input_dim, data2class_numbers, ratio_zoo
from typing import Union, TypeVar, List

TEMP_DIR = os.path.join(PROJECT_PATH, "script", ".temp_config")
if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)

T_path = TypeVar("T_path", str, Path)


def random_string(N=20):
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(N))


def is_true_iterator(value):
    if isinstance(value, Iterable):
        if not isinstance(value, (str, np.ndarray, torch.Tensor)):
            return True
    return False


def grid_search(**kwargs):
    max_N = 1
    for k, v in kwargs.copy().items():
        if is_true_iterator(v):
            max_N = max(max_N, len(v))
    for k, v in kwargs.copy().items():
        if is_true_iterator(v):
            kwargs[k] = iter(v)
        else:
            kwargs[k] = [v]
    for value in product(*kwargs.values()):
        yield dict(zip(kwargs.keys(), value))


class ScriptGenerator:

    def __init__(self, *, data_name, num_batches, save_dir, data_opt) -> None:
        super().__init__()
        self.conditions = []
        self._data_name = data_name
        self.conditions.append("Data.name={}".format(data_name))
        self._num_batches = num_batches
        self.conditions.append(f"Trainer.num_batches={num_batches}")
        self._save_dir = save_dir

    def grid_search_on(self, *, seed: int, **kwargs):
        pass

    @staticmethod
    def _get_hyper_param_string(**kwargs):
        def to_str(v):
            if isinstance(v, Iterable) and (not isinstance(v, str)):
                return "_".join([str(x) for x in v])
            return v

        list_string = [f"{k}_{to_str(v)}" for k, v in kwargs.items()]
        prefix = "/".join(list_string)
        return prefix


class BaselineGenerator(ScriptGenerator):

    def __init__(self, *, data_name, num_batches, max_epoch, save_dir, model_checkpoint=None, data_opt) -> None:
        super().__init__(data_name=data_name, num_batches=num_batches, save_dir=save_dir, data_opt=data_opt)
        self._model_checkpoint = model_checkpoint
        self.conditions.append(f"Arch.checkpoint={self._model_checkpoint or 'null'}")
        self._max_epoch = max_epoch
        self.conditions.append(f"Trainer.max_epoch={self._max_epoch}")

    def generate_single_script(self, save_dir, seed, labeled_scan_num):
        from semi_seg import ft_lr_zooms
        ft_lr = ft_lr_zooms[self._data_name]
        return f"python main.py Trainer.name=ft Trainer.save_dir={save_dir} " \
               f" Optim.lr={ft_lr:.7f}  RandomSeed={str(seed)} Data.labeled_scan_num={labeled_scan_num} " \
               f" {' '.join(self.conditions)}  "

    def grid_search_on(self, *, seed: Union[int, List[int]], **kwargs):
        jobs = []

        labeled_scan_list = ratio_zoo[self._data_name]

        for param in grid_search(**{**kwargs, **{"seed": seed}}):
            random_seed = param.pop("seed")
            sub_save_dir = self._get_hyper_param_string(**param)
            true_save_dir = os.path.join(self._save_dir, "Seed_" + str(random_seed), sub_save_dir)

            job = " && ".join(
                [self.generate_single_script(save_dir=os.path.join(true_save_dir, "tra", f"labeled_scan_{l:02d}"),
                                             seed=random_seed, labeled_scan_num=l)
                 for l in labeled_scan_list])

            jobs.append(job)
        return jobs


class PretrainScriptGenerator(ScriptGenerator):

    def __init__(self, *, data_name, num_batches, save_dir, pre_max_epoch, ft_max_epoch) -> None:
        super().__init__(data_name=data_name, num_batches=num_batches, save_dir=save_dir, )
        self._pre_max_epoch = pre_max_epoch
        self.conditions.append(f"Trainer.pre_max_epoch={pre_max_epoch}")
        self._ft_max_epoch = ft_max_epoch
        self.conditions.append(f"Trainer.ft_max_epoch={ft_max_epoch}")

    def get_hook_name(self):
        ...

    def get_hook_params(self, **kwargs):
        ...

    def generate_single_script(self, save_dir, seed, hook_path):
        from semi_seg import pre_lr_zooms, ft_lr_zooms
        pre_lr = pre_lr_zooms[self._data_name]
        ft_lr = ft_lr_zooms[self._data_name]
        return f"python pretrain_main.py Trainer.save_dir={save_dir} " \
               f" Optim.pre_lr={pre_lr:.7f} Optim.ft_lr={ft_lr:.7f} RandomSeed={str(seed)} " \
               f" {' '.join(self.conditions)}  " \
               f" --opt-path config/pretrain.yaml {hook_path}"


def move_dataset():
    if on_cc():
        from contrastyou import DATA_PATH
        return f" find {DATA_PATH}  " + "-name '*.zip' -exec cp {} $SLURM_TMPDIR \;"
    return ""
