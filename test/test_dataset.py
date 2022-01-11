from unittest import TestCase

from torch.utils.data import DataLoader

from contrastyou import DATA_PATH
from contrastyou.data import InfiniteRandomSampler, ScanBatchSampler
from semi_seg.augment import ACDCTransforms2
from semi_seg.data import ACDCDataset
from semi_seg.data.rearr import ContrastBatchSampler


class TestACDC(TestCase):
    def test_acdc_twice_transform(self):
        dataset = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=ACDCTransforms2().pretrain)
        (image1, image2, target1, target2), filename, meta = dataset[2]

    def test_acdc_single_transform(self):
        dataset = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=ACDCTransforms2().val)
        (image1, target1), filename, meta = dataset[2]

    def test_infinite_sampler(self):
        dataset = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=ACDCTransforms2().val)
        sampler = InfiniteRandomSampler(dataset, shuffle=False)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=4)
        for i, data in enumerate(dataloader):
            print(data[1:])
            if i == 10000:
                break

    def test_scan_sampler(self):
        dataset = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=ACDCTransforms2().val)
        batch_sampler = ScanBatchSampler(dataset, is_infinite=False, shuffle=True)
        for i in batch_sampler:
            print(i)

    def test_contrastive_sampler(self):
        dataset = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=ACDCTransforms2().val)
        batch_sampler = ContrastBatchSampler(dataset)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
        for (data), filenames, (partition, scan) in dataloader:
            print(filenames, partition, scan)
