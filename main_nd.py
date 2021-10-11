import numpy  # noqa

from main import main

if __name__ == '__main__':
    import torch

    # torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = True  # noqa
    main()
