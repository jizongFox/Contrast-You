import unittest

import torch

from contrastyou.arch import UNet, UNetFeatureMapEnum
from semi_seg.hooks.ccblock import CrossCorrelationHook


class TestCrossCorrelationHook(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.model = UNet(input_dim=1, num_classes=4, max_channel=1024)
        self.input_image = torch.randn(5, 1, 224, 224)

    def test_cc_hook(self):
        project_params = {"num_clusters": 20, "head_type": "mlp", "normalize": False, "num_subheads": 1,
                          "hidden_dim": 128}
        hook = CrossCorrelationHook(name="conv3", model=self.model, feature_name=UNetFeatureMapEnum.Conv3, cc_weight=0.1,
                                    kernel_size=3, projector_params=project_params)
        hook(unlabeled_image_tf=self.input_image, unlabeled_logits_tf=torch.zeros_like(self.input_image))
