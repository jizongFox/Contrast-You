# refactorize by jizong 2021-04-29
from .unet import *
from .unet2 import *


def get_arch(name: str, **kwargs):
    assert name.lower() in {"unet", "unet2"}, name
    if name.lower() == "unet":
        return UNet(**kwargs)
    return UNet2(**kwargs)
