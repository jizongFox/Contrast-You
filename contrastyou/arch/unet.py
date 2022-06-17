from collections import OrderedDict
from contextlib import contextmanager
from enum import Enum
from functools import lru_cache, partial

import torch
from loguru import logger
from torch import nn

from ._base import _Network, _check_params, _complete_arch_start2end
from .utils import get_requires_grad, get_bn_track

__all__ = ["UNet", "UNetFeatureMapEnum"]


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, momentum: float = 0.1):
        super(_ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class _UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, momentum=0.1):
        super(_UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module, _Network):
    layer_dimension = {"Conv1": 1, "Conv2": 2, "Conv3": 4, "Conv4": 8, "Conv5": 16, "Up_conv5": 8, "Up_conv4": 4,
                       "Up_conv3": 2, "Up_conv2": 1, "Deconv_1x1": None}

    encoder_names = ("Conv1", "Conv2", "Conv3", "Conv4", "Conv5")
    decoder_names = ("Up5", "Up_conv5", "Up4", "Up_conv4", "Up3", "Up_conv3", "Up2", "Up_conv2", "Deconv_1x1")
    arch_elements = encoder_names + decoder_names

    r"""the difference between layer_dimension and arch_elements is that we allow operations on layer_dimension 
    while the latter can server to intermediate usage, such as gradient stop"""

    def __init__(self, input_dim=3, num_classes=1, max_channel=256, momentum=0.1):
        super(UNet, self).__init__()
        self._input_dim = input_dim
        self._num_classes = num_classes
        assert max_channel % 16 == 0 and max_channel >= 128, max_channel
        self._max_channel = max_channel

        self._max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self._Conv1 = _ConvBlock(in_ch=input_dim, out_ch=self.get_channel_dim("Conv1"), momentum=momentum)
        self._Conv2 = _ConvBlock(in_ch=self.get_channel_dim("Conv1"), out_ch=self.get_channel_dim("Conv2"),
                                 momentum=momentum)
        self._Conv3 = _ConvBlock(in_ch=self.get_channel_dim("Conv2"), out_ch=self.get_channel_dim("Conv3"),
                                 momentum=momentum)
        self._Conv4 = _ConvBlock(in_ch=self.get_channel_dim("Conv3"), out_ch=self.get_channel_dim("Conv4"),
                                 momentum=momentum)
        self._Conv5 = _ConvBlock(in_ch=self.get_channel_dim("Conv4"), out_ch=self.get_channel_dim("Conv5"),
                                 momentum=momentum)

        self._Up5 = _UpConv(in_ch=self.get_channel_dim("Conv5"), out_ch=self.get_channel_dim("Up_conv5"),
                            momentum=momentum)
        self._Up_conv5 = _ConvBlock(in_ch=self.get_channel_dim("Conv5"), out_ch=self.get_channel_dim("Up_conv5"),
                                    momentum=momentum)

        self._Up4 = _UpConv(in_ch=self.get_channel_dim("Up_conv5"), out_ch=self.get_channel_dim("Up_conv4"),
                            momentum=momentum)
        self._Up_conv4 = _ConvBlock(in_ch=self.get_channel_dim("Up_conv5"), out_ch=self.get_channel_dim("Up_conv4"),
                                    momentum=momentum)

        self._Up3 = _UpConv(in_ch=self.get_channel_dim("Up_conv4"), out_ch=self.get_channel_dim("Up_conv3"),
                            momentum=momentum)
        self._Up_conv3 = _ConvBlock(in_ch=self.get_channel_dim("Up_conv4"), out_ch=self.get_channel_dim("Up_conv3"),
                                    momentum=momentum)

        self._Up2 = _UpConv(in_ch=self.get_channel_dim("Up_conv3"), out_ch=self.get_channel_dim("Up_conv2"),
                            momentum=momentum)
        self._Up_conv2 = _ConvBlock(in_ch=self.get_channel_dim("Up_conv3"), out_ch=self.get_channel_dim("Up_conv2"),
                                    momentum=momentum)

        self._Deconv_1x1 = nn.Conv2d(self.get_channel_dim("Up_conv2"), num_classes, kernel_size=(1, 1), stride=(1, 1),
                                     padding=(0, 0))

    def forward(self, x, until: str = None):
        if until and until not in self.layer_dimension:
            raise KeyError(f"`return_until` should be in {', '.join(self.layer_dimension.keys())},"
                           f" given {until}  ")
        # encoding path
        e1 = self._Conv1(x)  # 16 224 224
        # e1-> Conv1
        if until == "Conv1":
            return e1

        e2 = self._max_pool1(e1)
        e2 = self._Conv2(e2)  # 32 112 112
        # e2 -> Conv2
        if until == "Conv2":
            return e2

        e3 = self._max_pool2(e2)
        e3 = self._Conv3(e3)  # 64 56 56
        # e3->Conv3
        if until == "Conv3":
            return e3

        e4 = self._max_pool3(e3)
        e4 = self._Conv4(e4)  # 128 28 28
        # e4->Conv4
        if until == "Conv4":
            return e4

        e5 = self._max_pool4(e4)
        e5 = self._Conv5(e5)  # 256 14 14
        # e5->Conv5

        if until == "Conv5":
            return e5

        # decoding + concat path
        d5 = self._Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self._Up_conv5(d5)  # 128 28 28
        # d5->Up5+Up_conv5

        if until == "Up_conv5":
            return d5

        d4 = self._Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self._Up_conv4(d4)  # 64 56 56

        if until == "Up_conv4":
            return d4

        # d4->Up4+Up_conv4

        d3 = self._Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self._Up_conv3(d3)  # 32 112 112

        if until == "Up_conv3":
            return d3

        # d3->Up3+upconv3

        d2 = self._Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self._Up_conv2(d2)  # 16 224 224

        if until == "Up_conv2":
            return d2

        # d2->up2+upconv2
        d1 = self._Deconv_1x1(d2)  # 4 224 224
        return d1

    @lru_cache()
    def get_channel_dim(self, name: str):
        if name == "Deconv_1x1":
            return self._num_classes
        elif name in self.layer_dimension:
            return int(self.layer_dimension[name] / 16 * self._max_channel)
        else:
            raise KeyError(name)

    def get_module(self, name: str) -> nn.Module:
        assert name in self.arch_elements, name
        return getattr(self, f"_{name}")

    @contextmanager
    def switch_grad(self, enable=True, *, start: str = None, end: str = None, include_start=True, include_end=True):
        _check_params(start, end, include_start, include_end, model=self)
        start, end = (start or "Conv1"), (end or "Deconv_1x1")

        all_component = _complete_arch_start2end(start, end, include_start=include_start, include_end=include_end,
                                                 model=self)
        prev_state = OrderedDict()
        if len(all_component) > 0:
            logger.opt(depth=2).trace("set grad {} to {}", enable, ", ".join(all_component))
        for c in all_component:
            cur_module = getattr(self, f"_{c}")
            prev_state[c] = get_requires_grad(cur_module)
            cur_module.requires_grad_(enable)
        try:
            yield self
        finally:
            if len(all_component) > 0:
                logger.opt(depth=2).trace("restore previous status to {}", ", ".join(all_component))
            for c in all_component:
                cur_module = getattr(self, f"_{c}")
                cur_module.requires_grad_(prev_state[c])

    @contextmanager
    def switch_bn_track(self, enable=True, *, start: str = None, end: str = None, include_start=True, include_end=True):
        _check_params(start, end, include_start, include_end, model=self)
        start, end = (start or "Conv1"), (end or "Deconv_1x1")

        all_component = _complete_arch_start2end(start, end, include_start=include_start, include_end=include_end,
                                                 model=self)
        prev_state = OrderedDict()
        if len(all_component) > 0:
            logger.opt(depth=2).trace("set bn_track as {} to {}", enable, ", ".join(all_component))

        def switch_attr(m, enable=True):
            if hasattr(m, "track_running_stats"):
                m.track_running_stats = enable

        for c in all_component:
            cur_module = getattr(self, "_" + c)
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
                cur_module = getattr(self, "_" + c)
                cur_module.apply(partial(switch_attr, enable=prev_state[c]))

    @property
    def num_classes(self):
        return self._num_classes


class UNetFeatureMapEnum(Enum):
    Conv1 = "Conv1"
    Conv2 = "Conv2"
    Conv3 = "Conv3"
    Conv4 = "Conv4"
    Conv5 = "Conv5"
    Up_conv5 = "Up_conv5"
    Up_conv4 = "Up_conv4"
    Up_conv3 = "Up_conv3"
    Up_conv2 = "Up_conv2"
    Deconv_1x1 = "Deconv_1x1"
