from unittest import TestCase

from contrastyou.datasets import MMWHSDataset
from da_seg.augment import MMWHMStrongTransforms


class TestDataset(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._root = "./"

    def test_dataset(self):
        mr_dataset = MMWHSDataset(root_dir=self._root, modality="mr", mode="train",
                                  transforms=MMWHMStrongTransforms.pretrain)
        ct_dataset = MMWHSDataset(root_dir=self._root, modality="ct", mode="train",
                                  transforms=MMWHMStrongTransforms.pretrain)

        pass
