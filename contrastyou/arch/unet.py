import torch

from torch import nn
from torch.nn import functional as F

__all__ = ["UNet"]


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_dim=3, num_classes=1):
        super(UNet, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch=input_dim, out_ch=16)
        self.Conv2 = conv_block(in_ch=16, out_ch=32)
        self.Conv3 = conv_block(in_ch=32, out_ch=64)
        self.Conv4 = conv_block(in_ch=64, out_ch=128)
        self.Conv5 = conv_block(in_ch=128, out_ch=256)

        self.Up5 = up_conv(in_ch=256, out_ch=128)
        self.Up_conv5 = conv_block(in_ch=256, out_ch=128)

        self.Up4 = up_conv(in_ch=128, out_ch=64)
        self.Up_conv4 = conv_block(in_ch=128, out_ch=64)

        self.Up3 = up_conv(in_ch=64, out_ch=32)
        self.Up_conv3 = conv_block(in_ch=64, out_ch=32)

        self.Up2 = up_conv(in_ch=32, out_ch=16)
        self.Up_conv2 = conv_block(in_ch=32, out_ch=16)

        self.Conv_1x1 = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x, return_features=False):
        # encoding path
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # decoding + concat path
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        if return_features:
            return d1,(e5, e4, e3, e2, e1), (d5, d4, d3, d2)
        return d1

    def disable_grad_encoder(self):
        encoder_names = ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5"]
        self._set_grad(encoder_names, False)

    def enable_grad_encoder(self):
        encoder_names = ["Conv1", "Conv2", "Conv3", "Conv4", "Conv5"]
        self._set_grad(encoder_names, True)

    def disable_grad_decoder(self):
        decoder_names = ["Up5", "Up_conv5", "Up4", "Up_conv4", "Up3", "Up_conv3", "Up2", "Up_conv2", "Conv_1x1"]
        self._set_grad(decoder_names, False)

    def enable_grad_decoder(self):
        decoder_names = ["Up5", "Up_conv5", "Up4", "Up_conv4", "Up3", "Up_conv3", "Up2", "Up_conv2", "Conv_1x1"]
        self._set_grad(decoder_names, True)

    def _set_grad(self, name_list, requires_grad=False):
        for n in name_list:
            for parameter in getattr(self, n).parameters():
                parameter.requires_grad = requires_grad
