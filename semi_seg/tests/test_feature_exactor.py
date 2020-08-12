from unittest import TestCase

import torch

from contrastyou.arch import UNet
from semi_seg._utils import FeatureExtractor, LocalClusterWrappaer


class TestFeatureExtractor(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._net = UNet()
        self._image = torch.randn(1, 3, 224, 224)

    def test_feature_extractor(self):
        with FeatureExtractor(self._net,
                              ["Conv1", "Conv5", "Up_conv5", "Up_conv4", "DeConv_1x1"]) as feature_extractor:
            for i in range(3):
                segment, (e5, e4, e3, e2, e1), (d5, d4, d3, d2) = self._net(self._image, return_features=True)
                assert id(feature_extractor["Conv1"]) == id(e1)
                assert id(feature_extractor["Conv5"]) == id(e5)
                assert id(feature_extractor["Up_conv5"]) == id(d5)
                assert id(feature_extractor["DeConv_1x1"]) == id(segment)


class TestLocalClusterWrapper(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._net = UNet()
        self._image = torch.randn(1, 3, 224, 224)
        self._feature_names = ["Up_conv4", "Up_conv3", "Up_conv2"]
        self._num_subheads = [10, 20, 30]
        self._num_clusters = [11, 12, 13]

    def test_clusterwrapper(self):
        self._wrapper = LocalClusterWrappaer(self._feature_names, num_subheads=self._num_subheads,
                                             num_clusters=self._num_clusters)
        with FeatureExtractor(self._net, self._feature_names) as feature_extractor:
            segment = self._net(self._image)
            for i, (feature, projector) in enumerate(zip(feature_extractor, self._wrapper)):
                projectsimplex = projector(feature)
                assert len(projectsimplex) == self._num_subheads[i]
                assert projectsimplex[0].shape[1] == self._num_clusters[i]
