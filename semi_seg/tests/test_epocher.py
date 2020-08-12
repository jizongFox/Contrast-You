from unittest import TestCase

import torch
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.loss import KL_div
from deepclustering2.utils import set_benchmark
from torch import nn
from torch.utils.data import DataLoader

from contrastyou.arch import UNet
from contrastyou.losses.iic_loss import IIDSegmentationSmallPathLoss
from semi_seg._utils import LocalClusterWrappaer
from semi_seg.epocher import TrainEpocher, EvalEpocher, UDATrainEpocher, IICTrainEpocher, UDAIICEpocher
from semi_seg.tests._helper import create_acdc_dataset


class TestPartialEpocher(TestCase):

    def setUp(self) -> None:
        super().setUp()

        self.label_set, self.unlabel_set, self.val_set = create_acdc_dataset(0.1)

        self.labeled_loader = DataLoader(
            self.label_set, batch_size=2,
            sampler=InfiniteRandomSampler(self.label_set, shuffle=True),
        )
        self.unlabeled_loader = DataLoader(
            self.unlabel_set, batch_size=3,
            sampler=InfiniteRandomSampler(self.unlabel_set, shuffle=True)
        )
        self.val_loader = DataLoader(self.val_set, )
        self.net = UNet(input_dim=1, num_classes=4)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self._num_batches = 10
        set_benchmark(1)

    def test_partial_epocher(self):
        partial_trainer = TrainEpocher(self.net, self.optimizer, self.labeled_loader, self.unlabeled_loader,
                                       sup_criterion=KL_div(), reg_weight=0.0, num_batches=self._num_batches,
                                       cur_epoch=0,
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
                                      num_batches=self._num_batches, cur_epoch=0,
                                      device="cuda")
        uda_result = uda_trainer.run()
        print(uda_result)

    def test_iic_epocher(self):
        iic_segment_criterion = IIDSegmentationSmallPathLoss(padding=1, patch_size=64)
        feature_position = ["Up_conv3", "Up_conv2"]
        projectors_wrapper = LocalClusterWrappaer(feature_position, num_subheads=3, num_clusters=20).to("cuda")
        iic_epocher = IICTrainEpocher(self.net, projectors_wrapper=projectors_wrapper, optimizer=self.optimizer,
                                      labeled_loader=self.labeled_loader,
                                      unlabeled_loader=self.unlabeled_loader, sup_criterion=KL_div(),
                                      IIDSegCriterion=iic_segment_criterion, reg_weight=0.1,
                                      num_batches=self._num_batches, cur_epoch=0, device="cuda",
                                      feature_position=feature_position)
        result_dict = iic_epocher.run()
        print(result_dict)

    def test_udaiic_epocher(self):
        iic_segment_criterion = IIDSegmentationSmallPathLoss(padding=1, patch_size=64)
        feature_position = ["Up_conv3", "Up_conv2"]
        projectors_wrapper = LocalClusterWrappaer(feature_position, num_subheads=10, num_clusters=10).to("cuda")
        udaiic_epocher = UDAIICEpocher(self.net, projectors_wrapper=projectors_wrapper, optimizer=self.optimizer,
                                       labeled_loader=self.labeled_loader,
                                       unlabeled_loader=self.unlabeled_loader, sup_criterion=KL_div(),
                                       reg_criterion=nn.MSELoss(), IIDSegCriterion=iic_segment_criterion,
                                       reg_weight=1, num_batches=self._num_batches, cur_epoch=0, device="cuda",
                                       feature_position=feature_position, cons_weight=1, iic_weight=0.1)
        result_dict = udaiic_epocher.run()
        print(result_dict)
