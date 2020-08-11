from unittest import TestCase

import torch
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.loss import KL_div
from torch import nn
from torch.utils.data import DataLoader

from contrastyou.arch import UNet
from semi_seg.tests._helper import create_acdc_dataset
from semi_seg.trainer import TrainEpocher, EvalEpocher, UDATrainEpocher, IICTrainEpocher


class TestPartialEpocher(TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.label_set, self.unlabel_set, self.val_set = create_acdc_dataset(0.1)

        self.labeled_loader = DataLoader(
            self.label_set, batch_size=8,
            sampler=InfiniteRandomSampler(self.label_set, shuffle=True),
        )
        self.unlabeled_loader = DataLoader(
            self.unlabel_set, batch_size=3,
            sampler=InfiniteRandomSampler(self.unlabel_set, shuffle=True)
        )
        self.val_loader = DataLoader(self.val_set, )
        self.net = UNet(input_dim=1, num_classes=4)
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def test_partial_epocher(self):
        partial_trainer = TrainEpocher(self.net, self.optimizer, self.labeled_loader, self.unlabeled_loader,
                                       sup_criterion=KL_div(), reg_weight=0.0, num_batches=64, cur_epoch=0,
                                       device="cuda")
        train_result = partial_trainer.run()
        print(train_result)

    def test_val_epocher(self):
        val_trainer = EvalEpocher(self.net, sup_criterion=KL_div(), val_loader=self.val_loader, cur_epoch=0,
                                  device="cuda")
        val_result, cur_score = val_trainer.run()
        print(val_result)

    def test_uda_epocher(self):
        uda_trainer = UDATrainEpocher(self.net, self.optimizer, self.labeled_loader, self.unlabeled_loader,
                                      sup_criterion=KL_div(), reg_criterion=nn.MSELoss(), reg_weight=0.1,
                                      num_batches=64, cur_epoch=0,
                                      device="cuda")
        uda_result = uda_trainer.run()
        print(uda_result)

    def test_iic_epocher(self):
        iic_epocher = IICTrainEpocher(self.net, projectors_wrapper=None, optimizer=self.optimizer, labeled_loader=self.labeled_loader,
                                      unlabeled_loader=self.unlabeled_loader, sup_criterion=KL_div(), IIDSegCriterion=None, reg_weight=0.1,
                                      num_batches=64, cur_epoch=0, device="cuda")
        result_dict = iic_epocher.run()
