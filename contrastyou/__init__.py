from functools import lru_cache
from pathlib import Path

PROJECT_PATH = str(Path(__file__).parents[1])
DATA_PATH = str(Path(PROJECT_PATH) / ".data")
Path(DATA_PATH).mkdir(exist_ok=True, parents=True)
CONFIG_PATH = str(Path(PROJECT_PATH, "config"))


@lru_cache()
def get_cc_data_path():
    import os
    possible_path = os.environ.get("SLURM_TMPDIR", None)
    if possible_path:
        possible_folders = os.listdir(possible_path)
        if len(possible_folders) > 0:
            print("cc_data_path is {}".format(possible_path))
            return possible_path
    print("cc_data_path is {}".format(DATA_PATH))
    return DATA_PATH
