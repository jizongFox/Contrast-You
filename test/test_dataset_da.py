import typing
from unittest import TestCase

from easydict import EasyDict as edict

from utils import make_data_dataloaders

if typing.TYPE_CHECKING:
    pass
from semi_seg.data import get_data


class TestDADataset(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.data_params = edict({
            "name": "acdc",
            "labeled_scan_num": 1,
            "order_num": 0,
        })
        self.data_loader_params = edict({
            "shuffle": True,
            "batch_size": 5,
            "num_workers": 5,
        })

    def test_get_source_A_and_B_data(self):
        labeled_data, unlabeled_data, val_data, test_data = get_data(
            data_params=self.data_params,
            labeled_loader_params=self.data_loader_params, unlabeled_loader_params=self.data_loader_params,
            pretrain=False, total_freedom=False, order_num=0
        )
        unlabeled_data, val_data, test_data = make_data_dataloaders(unlabeled_data, val_data, test_data)

        for data in unlabeled_data:
            pass
