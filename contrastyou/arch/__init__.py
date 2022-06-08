# refactorize by jizong 2021-04-29
from loguru import logger

from .smp import UNet_SMP
from .unet import *
from .unet2 import *


def get_arch(name: str, **kwargs):
    assert name.lower() in {"unet", "unet2", "unetsmp"}, name
    if name.lower() == "unet":
        model = UNet(**kwargs)
    elif name.lower() == "unet2":
        model = UNet2(**kwargs)
    else:
        model = UNet_SMP(**kwargs)
    logger.info(f"Initializing {model.__class__.__name__}")

    return model
