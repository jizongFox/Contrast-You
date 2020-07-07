from enum import Enum
from pathlib import Path

PROJECT_PATH = str(Path(__file__).parents[1])
DATA_PATH = str(Path(PROJECT_PATH) / ".data")
Path(DATA_PATH).mkdir(exist_ok=True, parents=True)
CONFIG_PATH= str(Path(PROJECT_PATH, "config"))


class ModelState(Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"
    EVAL = "EVAL"

    @staticmethod
    def from_str(mode_str):
        """ Init from string
            :param mode_str: ['train', 'eval', 'predict']
        """
        if mode_str == "train":
            return ModelState.TRAIN
        elif mode_str == "test":
            return ModelState.TEST
        elif mode_str == "eval":
            return ModelState.EVAL
        else:
            raise ValueError("Invalid argument mode_str {}".format(mode_str))
