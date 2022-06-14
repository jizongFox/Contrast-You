# this is the adaptor for segmentation model pytorch
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache, partial
from typing import Optional, List, Union

import segmentation_models_pytorch as smp
import torch
from loguru import logger

from contrastyou.arch._base import _check_params, _complete_arch_start2end, _Network
from contrastyou.arch.utils import get_requires_grad, get_bn_track


class UNet_SMP(smp.Unet, _Network):
    encoder_names = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
    decoder_names = ("up5", "up4", "up3", "up2", "up1", "output")
    arch_elements = tuple(list(encoder_names) + list(decoder_names))

    def __init__(self, encoder_name: str = "resnet18", encoder_depth: int = 5,
                 encoder_weights: Optional[str] = "imagenet", decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (256, 128, 64, 32, 16), decoder_attention_type: Optional[str] = None,
                 input_dim: int = 3, num_classes: int = 2, activation: Optional[Union[str, callable]] = None,
                 aux_params: Optional[dict] = None, **kwargs):
        super().__init__(encoder_name, encoder_depth, encoder_weights, decoder_use_batchnorm, decoder_channels,
                         decoder_attention_type, input_dim, num_classes, activation, aux_params)
        self._num_classes = num_classes
        self.input_dim = input_dim
        self.max_channel = max(decoder_channels)
        self._decoder_channels = decoder_channels

    def get_module(self, name):
        assert name in self.arch_elements, f"{name} is not in {self.arch_elements}"
        if name in self.encoder_names:
            cur_module = getattr(self.encoder, f"{name}")
        elif name in self.decoder_names[:-1]:
            cur_module = self.decoder.blocks[self.decoder_names[:-1].index(name)]
        elif name in self.decoder_names[-1:]:
            cur_module = self.segmentation_head
        else:
            raise NotImplementedError(name)
        return cur_module

    @contextmanager
    def switch_grad(self, enable=True, *, start: str = None, end: str = None, include_start=True, include_end=True):
        _check_params(start, end, include_start, include_end, model=self)
        start, end = (start or "layer1"), (end or "output")

        all_component = _complete_arch_start2end(start, end, include_start=include_start, include_end=include_end,
                                                 model=self)
        prev_state = OrderedDict()
        if len(all_component) > 0:
            logger.opt(depth=2).trace("set grad {} to {}", enable, ", ".join(all_component))

        for c in all_component:
            cur_module = self.get_module(c)
            prev_state[c] = get_requires_grad(cur_module)
            cur_module.requires_grad_(enable)
        try:
            yield self
        finally:
            if len(all_component) > 0:
                logger.opt(depth=2).trace("restore previous status to {}", ", ".join(all_component))
            for c in all_component:
                cur_module = self.get_module(c)
                cur_module.requires_grad_(prev_state[c])

    @contextmanager
    def switch_bn_track(self, enable=True, *, start: str = None, end: str = None, include_start=True, include_end=True):
        _check_params(start, end, include_start, include_end, model=self)
        start, end = (start or "layer1"), (end or "output")

        all_component = _complete_arch_start2end(start, end, include_start=include_start, include_end=include_end,
                                                 model=self)
        prev_state = OrderedDict()
        if len(all_component) > 0:
            logger.opt(depth=2).trace("set bn_track as {} to {}", enable, ", ".join(all_component))

        def switch_attr(m, enable=True):
            if hasattr(m, "track_running_stats"):
                m.track_running_stats = enable

        for c in all_component:
            cur_module = self.get_module(c)
            try:
                prev_state[c] = get_bn_track(cur_module)
            except RuntimeError:
                continue
            cur_module.apply(partial(switch_attr, enable=enable))
        try:
            yield self
        finally:
            if len(all_component) > 0:
                logger.opt(depth=2).trace("restore previous states to {}", ", ".join(all_component))
            for c in prev_state.keys():
                cur_module = self.get_module(c)
                cur_module.apply(partial(switch_attr, enable=prev_state[c]))

    @property
    def num_classes(self):
        return self._num_classes

    @lru_cache()
    def get_channel_dim(self, name: str):
        if name in self.encoder_names:
            return self.encoder.out_channels[self.encoder_names.index(name)]
        elif name in self.decoder_names[:-1]:
            return self._decoder_channels[self.decoder_names.index(name)]
        elif name == "output":
            return self.num_classes
        else:
            raise KeyError(name)

    def forward(self, x, until=None, **kwargs):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


if __name__ == '__main__':
    model = UNet_SMP()
    input_ = torch.randn(1, 3, 256, 256)
    with model.switch_grad(False, start="layer1", include_start=False, end="output", include_end=False), \
            model.switch_bn_track(False, start="layer1", include_start=False, end="output", include_end=False):
        output = model(input_)
    output_ = model(input_)
    print(model)
