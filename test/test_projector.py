import unittest

import torch

from contrastyou.projectors import CrossCorrelationProjector


class TestCCProjector(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.feature = torch.randn(1, 10, 224, 224)

    def test_cc_projector(self):
        projector = CrossCorrelationProjector(input_dim=10, num_clusters=20, head_type="mlp", normalize=False,
                                              T=1.0, num_subheads=1, hidden_dim=64)
        distributions = projector(self.feature)
