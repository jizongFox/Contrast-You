from unittest import TestCase

import torch

from contrastyou.arch import UNet
from deepclustering2.loss import Entropy


class TestUnet(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self._img = torch.randn(10, 1, 224, 224)

    def test_unet(self):
        net = UNet(input_dim=1, num_classes=4)
        prediction = net(self._img, )
        prediction_representation = net(self._img, return_features=True)
        self.assertTrue(torch.allclose(prediction, prediction_representation[0]))
        self.assertFalse(id(prediction) == id(prediction_representation[0]))

    def test_encoder_grad(self):
        net = UNet(input_dim=1, num_classes=4)
        net.disable_grad_encoder()
        predict = net(self._img, return_features=True)[0]
        loss = Entropy()(predict.softmax(1))
        loss.backward()
        assert self._if_grad_disabled(net.Conv2.parameters().__next__().grad)
        assert not self._if_grad_disabled(net.Up_conv2.parameters().__next__().grad)

        net.zero_grad()
        net.enable_grad_encoder()
        net.disable_grad_decoder()
        predict = net(self._img, return_features=True)[0]
        loss = Entropy()(predict.softmax(1))
        loss.backward()
        assert not self._if_grad_disabled(net.Conv2.parameters().__next__().grad)
        assert self._if_grad_disabled(net.Up_conv2.parameters().__next__().grad)

    @staticmethod
    def _if_grad_disabled(grad):
        """check if a given grad is set to be None or Zerolike"""
        if grad is None:
            return True
        if torch.allclose(grad, torch.zeros_like(grad)):
            return True
        return False
