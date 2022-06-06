# this is the adaptor for segmentation model pytorch
from typing import Optional, List, Union

import segmentation_models_pytorch as smp


class UNet_SMP(smp.Unet):
    def __init__(self, encoder_name: str = "resnet18", encoder_depth: int = 5,
                 encoder_weights: Optional[str] = "imagenet", decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (256, 128, 64, 32, 16), decoder_attention_type: Optional[str] = None,
                 input_dim: int = 3, num_classes: int = 1, activation: Optional[Union[str, callable]] = None,
                 aux_params: Optional[dict] = None, **kwargs):
        super().__init__(encoder_name, encoder_depth, encoder_weights, decoder_use_batchnorm, decoder_channels,
                         decoder_attention_type, input_dim, num_classes, activation, aux_params)
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.max_channel = max(decoder_channels)
