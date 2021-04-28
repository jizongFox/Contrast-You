from pathlib import Path

from contrastyou.arch import _register_arch

PROJECT_PATH = str(Path(__file__).parents[1])
DATA_PATH = str(Path(PROJECT_PATH) / ".data")
Path(DATA_PATH).mkdir(exist_ok=True, parents=True)
CONFIG_PATH = str(Path(PROJECT_PATH, "config"))


def get_cc_data_path():
    import os
    possible_path = os.environ.get("SLURM_TMPDIR", None)
    if possible_path:
        possible_folders = os.listdir(possible_path)
        if len(possible_folders) > 0:
            return possible_path
    return DATA_PATH


_ = _register_arch
