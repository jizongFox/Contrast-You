# refactorize by jizong 2021-04-29
from .smp import UNet_SMP
from .unet import *
from .unet2 import *


def get_arch(name: str, **kwargs):
    assert name.lower() in {"unet", "unet2", "unetsmp"}, name
    if name.lower() == "unet":
        return UNet(**kwargs)
    if name.lower() == "unet2":
        return UNet2(**kwargs)

    return UNet_SMP(**kwargs)
