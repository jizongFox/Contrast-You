from unittest import TestCase

import torch
from torch.utils.data import DataLoader

from contrastyou.epocher._utils import preprocess_input_with_twice_transformation
from semi_seg.tests._helper import create_acdc_dataset


class TestACDCDataset(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.label_set, self.unlabel_set, self.val_set = create_acdc_dataset(0.1)

    def test_10_split(self):
        assert len(self.label_set.get_group_list()) == 174 // 10
        assert len(self.unlabel_set.get_group_list()) == (174 - 174 // 10)

    def test_100_split(self):
        label_set, unlabel_set, val_set = create_acdc_dataset(1.0)
        assert len(label_set.get_group_list()) == 174
        assert len(unlabel_set.get_group_list()) == 174

    def test_unfold_data(self):
        loader = DataLoader(self.label_set, batch_size=16)
        data = loader.__iter__().__next__()

        (image1, target1), (image2, target2), filename, partition, patient_group = \
            preprocess_input_with_twice_transformation(data, "cuda")
        assert image1.shape == torch.Size([16, 1, 224, 224])
        assert image1.shape == image2.shape
