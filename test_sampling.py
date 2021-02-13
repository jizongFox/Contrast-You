from unittest import TestCase

import torch
from torch import nn


class TestUpsampling(TestCase):
    def setUp(self) -> None:
        super().setUp()
        output_size = 10
        self._feature_map = torch.randn(1, 1, 100, 100, requires_grad=True)
        self._adaptive_avg = nn.AdaptiveAvgPool2d((output_size, output_size))
        self._adaptive_max = nn.AdaptiveMaxPool2d((output_size, output_size))
        self._bilinear = nn.UpsamplingBilinear2d(size=(output_size, output_size))

    def test_adaptive_sampling(self):
        feature_map = self._adaptive_avg(self._feature_map)
        b, c, h, w = feature_map.shape
        loss = (feature_map[:, :, h // 2, w // 2]).mean() ** 2
        loss.backward()
        import matplotlib.pyplot as plt
        plt.imshow(self._feature_map.grad.squeeze())
        plt.show()

    def test_up_sampling(self):
        feature_map = self._bilinear(self._feature_map)
        b, c, h, w = feature_map.shape
        loss = (feature_map[:, :, h // 2, w // 2]).mean() ** 2
        loss.backward()
        import matplotlib.pyplot as plt
        plt.imshow(self._feature_map.grad.squeeze())
        plt.show()
