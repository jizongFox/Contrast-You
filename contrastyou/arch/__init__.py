from deepclustering2.arch import _register_arch
from .unet import UNet, UNet_Index

_register_arch("ContrastUnet", UNet)
