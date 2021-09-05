import numpy  # noqa
from loguru import logger

from main import main

if __name__ == '__main__':
    import torch

    with logger.catch(reraise=True):
        # torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = True  # noqa
        main()
