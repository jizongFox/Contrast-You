import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from loguru import logger

PROJECT_PATH = str(Path(__file__).parents[1])  # absolute path

DATA_PATH = str(Path(PROJECT_PATH) / ".data")
MODEL_PATH = str(Path(PROJECT_PATH) / "runs")
CONFIG_PATH = str(Path(PROJECT_PATH, "config"))
OPT_PATH = str(Path(PROJECT_PATH, "opt"))
Path(DATA_PATH).mkdir(exist_ok=True, parents=True)
Path(MODEL_PATH).mkdir(exist_ok=True, parents=True)
Path(CONFIG_PATH).mkdir(exist_ok=True, parents=True)
Path(OPT_PATH).mkdir(exist_ok=True, parents=True)

__accounts = ["rrg-mpederso"]

logger_format = "<green>{time:MM/DD HH:mm:ss.SS}</green> | <level>{level: ^7}</level> |" \
                "{process.name:<5}.{thread.name:<5}: " \
                "<cyan>{name:<8}</cyan>:<cyan>{function:<10}</cyan>:<cyan>{line:<4}</cyan>" \
                " - <level>{message}</level>"


@lru_cache()
def config_logger():
    logger.remove()

    logger.add(sys.stderr, format=logger_format, backtrace=False, diagnose=True, colorize=True)


config_logger()


@lru_cache()
def get_git_hash_tag() -> str:
    try:
        _git_hash = subprocess.check_output(
            [f"cd {PROJECT_PATH}; git rev-parse HEAD"], shell=True
        ).strip().decode()
    except Exception as e:  # noqa
        logger.opt(exception=True).warning(e)
        _git_hash = "unknown_tag"

    return _git_hash


@lru_cache()
def get_git_timestamp():
    p = subprocess.Popen(["git", "log", '-1', '--date=iso'], stdout=subprocess.PIPE)
    out, err = p.communicate()
    m = re.search('\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', out.decode("utf-8"))
    try:
        date_time = m.group(0)
        _date, _time = date_time.strip().split(" ")
        # _date = _date.replace("-", "")
        _time = _time.replace(":", "-")
        date_time = _date + "_" + _time
    except Exception as e:
        logger.opt(exception=True).warning(e)
        date_time = "unknown-timestamp"
    return date_time


git_hash = get_git_hash_tag()[:11]
git_timestamp = get_git_timestamp()


@lru_cache()
def get_true_data_path() -> str:
    """get absolute path of data in CC."""
    possible_path = os.environ.get("SLURM_TMPDIR", None)
    if possible_path:
        possible_folders = os.listdir(possible_path)
        if len(possible_folders) > 0:
            logger.debug("true_data_path is {}".format(possible_path))
            return possible_path
    logger.debug("true_data_path is {}".format(DATA_PATH))
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


# import numpy to avoid some errors
import numpy as np  # noqa
