from unittest import TestCase

import torch
from torch.utils.data import DataLoader

from contrastyou.augment import AffineTensorTransform
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation
from semi_seg.tests._helper import create_acdc_dataset


class TestAffineTransform(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.label_set, self.unlabel_set, self.val_set = create_acdc_dataset(0.1)

        self.labeled_loader = DataLoader(self.label_set, batch_size=16)
        self._bilinear_transformer = AffineTensorTransform()
        self._nearest_transformer = AffineTensorTransform(mode="nearest")

    def test_affine_transform(self):
        data = self.labeled_loader.__iter__().__next__()
        (image, target), _, filename, partition, group = \
            preprocess_input_with_twice_transformation(data, "cuda")
        image_tf, affinematrix = self._bilinear_transformer(image, independent=True)
        target_tf, _ = self._nearest_transformer(target.float(), AffineMatrix=affinematrix)

        assert torch.allclose(target_tf.float().unique(), target.float().unique())
