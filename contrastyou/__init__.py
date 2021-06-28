import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional

from loguru import logger

PROJECT_PATH = str(Path(__file__).parents[1])  # absolute path
DATA_PATH = str(Path(PROJECT_PATH) / ".data")
MODEL_PATH = str(Path(PROJECT_PATH) / "runs")
CONFIG_PATH = str(Path(PROJECT_PATH, "config"))

Path(DATA_PATH).mkdir(exist_ok=True, parents=True)
Path(MODEL_PATH).mkdir(exist_ok=True, parents=True)
Path(CONFIG_PATH).mkdir(exist_ok=True, parents=True)


@lru_cache()
def get_git_hash() -> Optional[str]:
    try:
        git_hash = subprocess.check_output(
            [f"cd {PROJECT_PATH}; git rev-parse HEAD"], shell=True
        ).strip().decode()
    except:  # noqa
        git_hash = None

    return git_hash


@lru_cache()
def get_cc_data_path() -> str:
    """get absolute path of data in CC."""
    possible_path = os.environ.get("SLURM_TMPDIR", None)
    if possible_path:
        possible_folders = os.listdir(possible_path)
        if len(possible_folders) > 0:
            print("cc_data_path is {}".format(possible_path))
            return possible_path
    logger.debug("cc_data_path is {}".format(DATA_PATH))
    return DATA_PATH


def on_cc() -> bool:
    """return if running on Compute Canada"""
    import socket
    hostname = socket.gethostname()
    # on beluga
    if "beluga" in hostname or "blg" in hostname:
        return True
    # on cedar
    if "cedar" in hostname or "cdr" in hostname:
        return True
    # on graham
    if "gra" in hostname:
        return True
    return False


def success(save_dir: str):
    filename = ".success"
    Path(str(save_dir), filename).touch()
