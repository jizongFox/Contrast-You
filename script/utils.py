import os
import random
import string
from collections import Iterable
from itertools import product
from pathlib import Path, PosixPath
from pprint import pprint
from typing import Union, Dict, Any, TypeVar

import yaml

from contrastyou import PROJECT_PATH, on_cc
from semi_seg import data2input_dim, data2class_numbers

T_path = TypeVar("T_path", str, Path)

TEMP_DIR = os.path.join(PROJECT_PATH, "script", ".temp_config")
if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)


def path2Path(path: T_path) -> Path:
    assert isinstance(path, (Path, str)), type(path)
    return Path(path) if isinstance(path, str) else path


def path2str(path: T_path) -> str:
    assert isinstance(path, (Path, str)), type(path)
    return str(path)


def check_hook_name(name):
    assert name in ("infonce", "spinfonce")


def random_string(N=20):
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(N))


def write_yaml(
    dictionary: Dict, save_dir: Union[Path, str], save_name: str, force_overwrite=True
) -> str:
    save_path = path2Path(save_dir) / save_name
    path2Path(save_dir).mkdir(exist_ok=True, parents=True)
    if save_path.exists():
        if force_overwrite is False:
            save_name = (
                save_name.split(".")[0] + "_copy" + "." + save_name.split(".")[1]
            )
    with open(str(save_path), "w") as outfile:  # type: ignore
        yaml.dump(dictionary, outfile, default_flow_style=False)
    return str(save_path)


def yaml_load(yaml_path: Union[Path, str], verbose=False) -> Dict[str, Any]:
    """
    load yaml file given a file string-like file path. return must be a dictionary.
    :param yaml_path:
    :param verbose:
    :return:
    """
    assert isinstance(yaml_path, (Path, str, PosixPath)), type(yaml_path)
    with open(path2str(yaml_path), "r") as stream:
        data_loaded: dict = yaml.safe_load(stream)
    if verbose:
        print(f"Loaded yaml path:{str(yaml_path)}")
        pprint(data_loaded)
    return data_loaded


def is_true_iterator(value):
    if isinstance(value, Iterable):
        if not isinstance(value, str):
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

    def __init__(self, *, data_name, num_batches, save_dir) -> None:
        super().__init__()
        self.conditions = []
        self._data_name = data_name
        self.conditions.append("Data.name={}".format(data_name))
        self._num_batches = num_batches
        self.conditions.append(f"Trainer.num_batches={num_batches}")
        self._save_dir = save_dir
        self._input_dim = data2input_dim[data_name]
        self.conditions.append(f"Arch.input_dim={self._input_dim}")
        self._num_classes = data2class_numbers[data_name]
        self.conditions.append(f"Arch.num_classes={self._num_classes}")

    def grid_search_on(self, *, seed: int, **kwargs):
        pass

    def get_hyparam_string(self, **kwargs):
        def to_str(v):
            if isinstance(v, Iterable) and (not isinstance(v, str)):
                return "_".join([str(x) for x in v])
            return v

        list_string = [f"{k}_{to_str(v)}" for k, v in kwargs.items()]
        prefix = "/".join(list_string)
        return prefix


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
