import itertools
import os
from pathlib import Path

import torch
from torch import nn

from contrastyou import PROJECT_PATH
from contrastyou.epocher.IIC_epocher import IICPretrainEcoderEpoch, IICPretrainDecoderEpoch
from contrastyou.losses.contrast_loss import SupConLoss
from contrastyou.trainer._utils import Flatten
from contrastyou.trainer.contrast_trainer import ContrastTrainer
from deepclustering2.meters2 import StorageIncomeDict
from deepclustering2.schedulers import GradualWarmupScheduler


class IICContrastTrainer(ContrastTrainer):
    RUN_PATH = Path(PROJECT_PATH) / "runs"

    def pretrain_encoder_init(self, group_option, num_clusters=40, iic_weight=1):
        self._projector_contrastive = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 256),
        )
        self._projector_iic = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, num_clusters),
            nn.Softmax(1)
        )
        self._optimizer = torch.optim.Adam(
            itertools.chain(self._model.parameters(), self._projector_contrastive.parameters(),
                            self._projector_iic.parameters()), lr=1e-6, weight_decay=1e-5)  # noqa
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer,
                                                                     self._max_epoch_train_encoder - 10, 0)
        self._scheduler = GradualWarmupScheduler(self._optimizer, 300, 10, self._scheduler)  # noqa

        self._group_option = group_option  # noqa

        # set augmentation method as `total_freedom = True`
        self._pretrain_loader.dataset._transform._total_freedom = True  # noqa
        self._pretrain_loader_iter = iter(self._pretrain_loader)  # noqa
        self._iic_weight = iic_weight

    def pretrain_encoder_run(self):
        self.to(self._device)
        self._model.enable_grad_encoder()  # noqa
        self._model.disable_grad_decoder()  # noqa

        for self._cur_epoch in range(self._start_epoch, self._max_epoch_train_encoder):
            pretrain_encoder_dict = IICPretrainEcoderEpoch(
                model=self._model, projection_head=self._projector_contrastive,
                projection_classifier=self._projector_iic, optimizer=self._optimizer,
                pretrain_encoder_loader=self._pretrain_loader_iter, contrastive_criterion=SupConLoss(),
                num_batches=self._num_batches, cur_epoch=self._cur_epoch, device=self._device,
                group_option=self._group_option, iic_weight_ratio=self._iic_weight,
            ).run()
            self._scheduler.step()
            storage_dict = StorageIncomeDict(PRETRAIN_ENCODER=pretrain_encoder_dict, )
            self._pretrain_encoder_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_dict, self._cur_epoch)
            self._save_to("last.pth", path=os.path.join(self._save_dir, "pretrain_encoder"))
        self.train_encoder_done = True

    # def pretrain_decoder_init(self, *args, **kwargs):
    #     # adding optimizer and scheduler
    #     self._projector_contrastive = nn.Sequential(
    #         nn.Conv2d(64, 64, 3, 1, 1),
    #         nn.BatchNorm2d(64),
    #         nn.LeakyReLU(0.01, inplace=True),
    #         nn.Conv2d(64, 32, 3, 1, 1)
    #     )
    #     self._proejctor_iic = None
    #     self._optimizer = torch.optim.Adam(itertools.chain(self._model.parameters(), self._projector.parameters()),
    #                                        lr=1e-6, weight_decay=0)
    #     self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer,
    #                                                                  self._max_epoch_train_decoder - 10, 0)
    #     self._scheduler = GradualWarmupScheduler(self._optimizer, 300, 10, self._scheduler)
    #
    #     # set augmentation method as `total_freedom = False`
    #     self._pretrain_loader.dataset._transform._total_freedom = False  # noqa
    #     self._pretrain_loader_iter = iter(self._pretrain_loader)  # noqa
    #
    # def pretrain_decoder_run(self):
    #     self.to(self._device)
    #     self._projector.to(self._device)
    #
    #     self._model.enable_grad_decoder()  # noqa
    #     self._model.disable_grad_encoder()  # noqa
    #
    #     for self._cur_epoch in range(self._start_epoch, self._max_epoch_train_decoder):
    #         pretrain_decoder_dict = IICPretrainDecoderEpoch(
    #             model=self._model, projection_head=self._projector, projection_classifier=self._projector_iic,
    #             optimizer=self._optimizer,
    #             pretrain_decoder_loader=self._pretrain_loader_iter,
    #             contrastive_criterion=SupConLoss(), num_batches=self._num_batches,
    #             cur_epoch=self._cur_epoch, device=self._device,
    #         ).run()
    #         self._scheduler.step()
    #         storage_dict = StorageIncomeDict(PRETRAIN_DECODER=pretrain_decoder_dict, )
    #         self._pretrain_encoder_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
    #         self._writer.add_scalar_with_StorageDict(storage_dict, self._cur_epoch)
    #         self._save_to("last.pth", path=os.path.join(self._save_dir, "pretrain_decoder"))
    #     self.train_decoder_done = True
