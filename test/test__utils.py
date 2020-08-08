from unittest import TestCase

import torch
from torch.nn import functional as F

from contrastyou.trainer._utils import LocalProjectionHead


class TestLocalProjectionHead(TestCase):

    def test_localprojector(self):
        projector = LocalProjectionHead(input_dim=10, output_size=(3, 3), head_type="mlp")
        features = torch.randn(1, 10, 256, 256, requires_grad=True)
        out = projector(features)
        out.retain_grad()
        loss = out[:, :, 1, 1].sum()
        loss.backward()
        import matplotlib.pyplot as plt
        plt.subplot(121)
        plt.imshow(features.grad[0].sum(0))
        plt.subplot(122)
        plt.imshow(out.grad[0, 0])
        plt.show()

