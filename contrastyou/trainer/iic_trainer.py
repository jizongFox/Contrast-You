import itertools
import os

import torch
from torch import nn

from contrastyou.epocher.IIC_epocher import IICPretrainEcoderEpoch, IICPretrainDecoderEpoch
from contrastyou.losses.contrast_loss import SupConLoss
from contrastyou.trainer._utils import SoftmaxWithT, ClassifierHead, ProjectionHead
from contrastyou.trainer.contrast_trainer import ContrastTrainer
from deepclustering2.meters2 import StorageIncomeDict
from deepclustering2.schedulers import GradualWarmupScheduler


class IICContrastTrainer(ContrastTrainer):

    def pretrain_encoder_init(self, group_option: str, lr=1e-6, weight_decay=1e-5, multiplier=300, warmup_max=10,
                              num_clusters=20, num_subheads=10, iic_weight=1, disable_contrastive=False, ctemperature=1,
                              ctype="linear", ptype="mlp"):
        self._projector_contrastive = ProjectionHead(input_dim=256, output_dim=256, head_type=ptype)
        self._projector_iic = ClassifierHead(
            input_dim=256, num_clusters=num_clusters, head_type=ctype, T=ctemperature, num_subheads=num_subheads
        )
        self._optimizer = torch.optim.Adam(
            itertools.chain(self._model.parameters(), self._projector_contrastive.parameters(),
                            self._projector_iic.parameters()), lr=lr, weight_decay=weight_decay)  # noqa
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer,
                                                                     self._max_epoch_train_encoder - warmup_max, 0)
        self._scheduler = GradualWarmupScheduler(self._optimizer, multiplier, warmup_max, self._scheduler)  # noqa

        self._group_option = group_option  # noqa
        self._disable_contrastive = disable_contrastive

        # set augmentation method as `total_freedom = True`
        assert hasattr(self._pretrain_loader.dataset._transform, "_total_freedom")  # noqa
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
                group_option=self._group_option, iic_weight=self._iic_weight,
                disable_contrastive=self._disable_contrastive
            ).run()
            self._scheduler.step()
            storage_dict = StorageIncomeDict(PRETRAIN_ENCODER=pretrain_encoder_dict, )
            self._pretrain_encoder_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_dict, self._cur_epoch)
            self._save_to("last.pth", path=os.path.join(self._save_dir, "pretrain_encoder"))
        self.train_encoder_done = True

    def pretrain_decoder_init(self, lr: float = 1e-6, weight_decay: float = 0.0, multiplier: int = 300, warmup_max=10,
                              num_clusters=20, temperature=10):
        # adding optimizer and scheduler
        self._projector_contrastive = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1)
        )
        self._projector_iic = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(64, num_clusters, 3, 1, 1),
            SoftmaxWithT(1, T=temperature)
        )
        self._optimizer = torch.optim.Adam(itertools.chain(
            self._model.parameters(),
            self._projector_contrastive.parameters(),
            self._projector_iic.parameters(),
        ),
            lr=lr, weight_decay=weight_decay
        )
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer,
                                                                     self._max_epoch_train_decoder - warmup_max, 0)
        self._scheduler = GradualWarmupScheduler(self._optimizer, multiplier, warmup_max, self._scheduler)

        # set augmentation method as `total_freedom = False`
        assert hasattr(self._pretrain_loader.dataset._transform, "_total_freedom")  # noqa
        self._pretrain_loader.dataset._transform._total_freedom = False  # noqa
        self._pretrain_loader_iter = iter(self._pretrain_loader)  # noqa

    def pretrain_decoder_run(self):
        self.to(self._device)

        self._model.enable_grad_decoder()  # noqa
        self._model.disable_grad_encoder()  # noqa

        for self._cur_epoch in range(self._start_epoch, self._max_epoch_train_decoder):
            pretrain_decoder_dict = IICPretrainDecoderEpoch(
                model=self._model, projection_head=self._projector_contrastive,
                projection_classifier=self._projector_iic,
                optimizer=self._optimizer,
                pretrain_decoder_loader=self._pretrain_loader_iter,
                contrastive_criterion=SupConLoss(), num_batches=self._num_batches,
                cur_epoch=self._cur_epoch, device=self._device,
            ).run()
            self._scheduler.step()
            storage_dict = StorageIncomeDict(PRETRAIN_DECODER=pretrain_decoder_dict, )
            self._pretrain_encoder_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_dict, self._cur_epoch)
            self._save_to("last.pth", path=os.path.join(self._save_dir, "pretrain_decoder"))
        self.train_decoder_done = True
