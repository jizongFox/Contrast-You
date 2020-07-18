import itertools
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from contrastyou import PROJECT_PATH
from contrastyou.epocher.base_epocher import FSEpocher
from contrastyou.epocher.contrast_epocher import PretrainEncoderEpoch, PretrainDecoderEpoch, FineTuneEpoch
from contrastyou.losses.contrast_loss import SupConLoss
from contrastyou.trainer._utils import Flatten
from deepclustering2.loss import KL_div
from deepclustering2.meters2 import Storage, StorageIncomeDict
from deepclustering2.schedulers import GradualWarmupScheduler
from deepclustering2.trainer.trainer import Trainer, T_loader
from deepclustering2.writer import SummaryWriter


class ContrastTrainer(Trainer):
    RUN_PATH = Path(PROJECT_PATH) / "runs"

    def __init__(self, model: nn.Module, pretrain_loader: T_loader, fine_tune_loader: T_loader, val_loader: DataLoader,
                 save_dir: str = "base", max_epoch_train_encoder: int = 100, max_epoch_train_decoder: int = 100,
                 max_epoch_train_finetune: int = 100, num_batches: int = 256, device: str = "cpu", configuration=None):
        """
        ContrastTraining Trainer
        :param model: nn.module network to be pretrained
        :param pretrain_loader: all unlabeled data under ContrastiveBatchSampler
        :param fine_tune_loader: a fraction of labeled data for finetuning
        :param val_loader: validation data
        :param save_dir: main save_die
        :param max_epoch_train_encoder: max_epoch to be trained for encoder training
        :param max_epoch_train_decoder: max_epoch to be trained for decoder training
        :param max_epoch_train_finetune: max_epoch to be trained for finetuning
        :param num_batches:
        :param device: cpu or cuda
        :param configuration: configuration dict
        """
        super().__init__(model, save_dir, None, num_batches, device, configuration)  # noqa
        self._pretrain_loader = pretrain_loader
        self._fine_tune_loader = fine_tune_loader
        self._val_loader = val_loader

        self._max_epoch_train_encoder = max_epoch_train_encoder
        self._max_epoch_train_decoder = max_epoch_train_decoder
        self._max_epoch_train_finetune = max_epoch_train_finetune

    def _run_pretrain_encoder(self):
        # adding optimizer and scheduler
        projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 128),
        )
        optimizer = torch.optim.Adam(itertools.chain(self._model.parameters(), projector.parameters()),
                                     lr=1e-6, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self._max_epoch_train_encoder - 10, 0)
        scheduler = GradualWarmupScheduler(optimizer, 300, 10, scheduler)
        self.to(self._device)
        projector.to(self._device)

        with Storage() as self._pretrain_encoder_storage:
            for self._cur_epoch in range(self._start_epoch, self._max_epoch_train_encoder):
                pretrain_encoder_dict = PretrainEncoderEpoch(
                    model=self._model, projection_head=projector,
                    optimizer=optimizer,
                    pretrain_encoder_loader=self._pretrain_loader,
                    contrastive_criterion=SupConLoss(), num_batches=self._num_batches,
                    cur_epoch=self._cur_epoch, device=self._device
                ).run()
                scheduler.step()
                storage_dict = StorageIncomeDict(pretrain_encode=pretrain_encoder_dict, )
                self._pretrain_encoder_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
                for k, v in storage_dict.__dict__.items():
                    self._writer.add_scalar_with_tag(k, v, global_step=self._cur_epoch)

    def _run_pretrain_decoder(self):
        # adding optimizer and scheduler
        projector = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, 1, 1)
        )
        optimizer = torch.optim.Adam(itertools.chain(self._model.parameters(), projector.parameters()),
                                     lr=1e-6, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self._max_epoch_train_decoder - 10, 0)
        scheduler = GradualWarmupScheduler(optimizer, 300, 10, scheduler)

        self.to(self._device)
        projector.to(self._device)
        with Storage() as self._pretrain_decoder_storage:
            for self._cur_epoch in range(self._start_epoch, self._max_epoch_train_decoder):
                pretrain_decoder_dict = PretrainDecoderEpoch(
                    model=self._model, projection_head=projector,
                    optimizer=optimizer,
                    pretrain_decoder_loader=self._pretrain_loader,
                    contrastive_criterion=SupConLoss(), num_batches=self._num_batches,
                    cur_epoch=self._cur_epoch, device=self._device
                ).run()
                scheduler.step()
                storage_dict = StorageIncomeDict(pretrain_decode=pretrain_decoder_dict, )
                self._pretrain_encoder_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
                for k, v in storage_dict.__dict__.items():
                    self._writer.add_scalar_with_tag(k, v, global_step=self._cur_epoch)

    def _run_finetune(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-6, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self._max_epoch_train_decoder - 10, 0)
        scheduler = GradualWarmupScheduler(optimizer, 300, 10, scheduler)

        self.to(self._device)

        with Storage() as self._finetune_storage:
            for self._cur_epoch in range(self._start_epoch, self._max_epoch_train_finetune):
                finetune_dict = FineTuneEpoch(
                    model=self._model, optimizer=optimizer,
                    labeled_loader=self._fine_tune_loader, num_batches=self._num_batches,
                    cur_epoch=self._cur_epoch, device=self._device
                ).run()
                val_dict, cur_score = FSEpocher.EvalEpoch(self._model, val_data_loader=self._val_loader,
                                                          sup_criterion=KL_div(),
                                                          cur_epoch=self._cur_epoch, device=self._device).run()
                scheduler.step()
                storage_dict = StorageIncomeDict(finetune=finetune_dict, val=val_dict)
                self._finetune_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
                for k, v in storage_dict.__dict__.items():
                    self._writer.add_scalar_with_tag(k, v, global_step=self._cur_epoch)

                self.save(cur_score)

    def start_training(self):
        with SummaryWriter(str(self._save_dir)) as self._writer:
            self._run_pretrain_encoder()
            self._run_pretrain_decoder()
            self._run_finetune()
