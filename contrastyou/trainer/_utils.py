import contextlib
import os
import shutil
from pathlib import Path
from random import Random as _Random
from typing import Union

import torch
from loguru import logger


class _RandomNameSequence:
    """An instance of _RandomNameSequence generates an endless
    sequence of unpredictable strings which can safely be incorporated
    into file names.  Each string is eight characters long.  Multiple
    threads can safely use the same instance at the same time.

    _RandomNameSequence is an iterator."""

    characters = "abcdefghijklmnopqrstuvwxyz0123456789_"

    @property
    def rng(self):
        cur_pid = os.getpid()
        if cur_pid != getattr(self, '_rng_pid', None):
            self._rng = _Random()
            self._rng_pid = cur_pid
        return self._rng

    def __iter__(self):
        return self

    def __next__(self):
        c = self.characters
        choose = self.rng.choice
        letters = [choose(c) for _ in range(8)]
        return ''.join(letters)


random_name = _RandomNameSequence()


def safe_save(checkpoint_dictionary, save_path):
    tmp_name = f"/tmp/{next(random_name)}"
    try:
        torch.save(checkpoint_dictionary, tmp_name)
        shutil.move(tmp_name, str(save_path))
    except Exception as e:
        logger.error(e)


@contextlib.contextmanager
def create_save_dir(self, save_dir: Union[Path, str]):
    """
    return absolute path given a save_dir
    if save_dir is a relative path, return MODEL_PATH/SAVE_DIR
    if save_dir is an absolute path, return this path
    """
    save_dir = str(save_dir)
    if not Path(save_dir).is_absolute():
        save_dir = str(Path(self.RUN_PATH) / save_dir)  # absolute path
        logger.trace(f"relative path found, set path to {save_dir}")
    yield save_dir
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    assert os.path.isabs(save_dir), f"save_dir must be an absolute path, given {save_dir}."
    return save_dir


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
        else:
            raise RuntimeError(f"{f} has been called more than once.")

    wrapper.has_run = False
    return wrapper
