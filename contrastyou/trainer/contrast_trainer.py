import itertools
import os
from pathlib import Path

import torch
from contrastyou import PROJECT_PATH
from contrastyou.arch import UNet, UNetFeatureExtractor
from contrastyou.epocher import PretrainEncoderEpoch, PretrainDecoderEpoch, SimpleFineTuneEpoch, MeanTeacherEpocher
from contrastyou.epocher.base_epocher import EvalEpoch
from contrastyou.losses.contrast_loss import SupConLoss
from contrastyou.trainer._utils import ProjectionHead
from deepclustering2.loss import KL_div
from deepclustering2.meters2 import Storage, StorageIncomeDict
from deepclustering2.schedulers import GradualWarmupScheduler
from deepclustering2.trainer.trainer import Trainer, T_loader
from deepclustering2.writer import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader


class ContrastTrainer(Trainer):
    RUN_PATH = Path(PROJECT_PATH) / "runs"

    def __init__(self, model: nn.Module, pretrain_loader: T_loader, fine_tune_loader: T_loader, val_loader: DataLoader,
                 save_dir: str = "base", max_epoch_train_encoder: int = 100, max_epoch_train_decoder: int = 100,
                 max_epoch_train_finetune: int = 100, num_batches: int = 256, device: str = "cpu", configuration=None,
                 train_encoder: bool = True, train_decoder: bool = True):
        """
        ContrastTraining Trainer
        :param model: nn.module network to be pretrained
        :param pretrain_loader: all unlabeled data under ContrastiveBatchSampler
        :param fine_tune_loader: a fraction of labeled data for finetuning, with InfiniteSampler
        :param val_loader: validation data
        :param save_dir: main save_die
        :param max_epoch_train_encoder: max_epoch to be trained for encoder training
        :param max_epoch_train_decoder: max_epoch to be trained for decoder training
        :param max_epoch_train_finetune: max_epoch to be trained for finetuning
        :param num_batches:  num_batches used in training
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

        self._register_buffer("train_encoder", train_encoder)
        self._register_buffer("train_decoder", train_decoder)
        self._register_buffer("train_encoder_done", False)
        self._register_buffer("train_decoder_done", False)

        self._pretrain_encoder_storage = Storage(csv_save_dir=os.path.join(self._save_dir, "pretrain_encoder"),
                                                 csv_name="encoder.csv")
        self._pretrain_decoder_storage = Storage(csv_save_dir=os.path.join(self._save_dir, "pretrain_decoder"),
                                                 csv_name="decoder.csv")
        self._finetune_storage = Storage(csv_save_dir=os.path.join(self._save_dir, "finetune"), csv_name="finetune.csv")

        # place holder for optimizer and scheduler
        self._optimizer = None
        self._scheduler = None
        self._projector = None
        self._sup_criterion = None

    def pretrain_encoder_init(self, group_option: str, lr=1e-6, weight_decay=1e-5, multiplier=300, warmup_max=10,
                              ptype="mlp"):
        # adding optimizer and scheduler
        self._projector = ProjectionHead(input_dim=256, output_dim=256, head_type=ptype)  # noqa
        self._optimizer = torch.optim.Adam(
            itertools.chain(self._model.parameters(),  # noqa
                            self._projector.parameters()),  # noqa
            lr=lr, weight_decay=weight_decay
        )  # noqa
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,  # noqa
            self._max_epoch_train_encoder - warmup_max, 0
        )
        self._scheduler = GradualWarmupScheduler(self._optimizer, multiplier, warmup_max, self._scheduler)  # noqa

        self._group_option = group_option  # noqa

        # set augmentation method as `total_freedom = True`
        assert hasattr(self._pretrain_loader.dataset._transform, "_total_freedom")  # noqa
        self._pretrain_loader.dataset._transform._total_freedom = True  # noqa
        self._pretrain_loader_iter = iter(self._pretrain_loader)  # noqa

    def pretrain_encoder_run(self):
        self.to(self._device)
        self._model.enable_grad_encoder()  # noqa
        self._model.disable_grad_decoder()  # noqa

        for self._cur_epoch in range(self._start_epoch, self._max_epoch_train_encoder):
            pretrain_encoder_dict = PretrainEncoderEpoch(
                model=self._model, projection_head=self._projector,
                optimizer=self._optimizer,
                pretrain_encoder_loader=self._pretrain_loader_iter,
                contrastive_criterion=SupConLoss(), num_batches=self._num_batches,
                cur_epoch=self._cur_epoch, device=self._device, group_option=self._group_option
            ).run()
            self._scheduler.step()
            storage_dict = StorageIncomeDict(PRETRAIN_ENCODER=pretrain_encoder_dict, )
            self._pretrain_encoder_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_dict, self._cur_epoch)
            self._save_to("last.pth", path=os.path.join(self._save_dir, "pretrain_encoder"))
        self.train_encoder_done = True

    def pretrain_decoder_init(self, lr: float = 1e-6, weight_decay: float = 0.0, multiplier: int = 300, warmup_max=10,
                              extract_postiion="Up_conv3", disable_grad_encoder=True):
        # feature_exactor
        self._extract_position = extract_postiion
        self._feature_extractor = UNetFeatureExtractor(self._extract_position)
        projector_input_dim = UNet.dimension_dict[extract_postiion]
        # if disable_encoder's gradient
        self._disable_grad_encoder = disable_grad_encoder

        # adding optimizer and scheduler
        self._projector = nn.Sequential(
            nn.Conv2d(projector_input_dim, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.AdaptiveAvgPool2d((4, 4))
        )  # fixme
        self._optimizer = torch.optim.Adam(itertools.chain(self._model.parameters(), self._projector.parameters()),
                                           lr=lr, weight_decay=weight_decay)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer,
                                                                     self._max_epoch_train_decoder - warmup_max, 0)
        self._scheduler = GradualWarmupScheduler(self._optimizer, multiplier, warmup_max, self._scheduler)

        # set augmentation method as `total_freedom = False`
        assert hasattr(self._pretrain_loader.dataset._transform, "_total_freedom")  # noqa
        self._pretrain_loader.dataset._transform._total_freedom = False  # noqa
        self._pretrain_loader_iter = iter(self._pretrain_loader)  # noqa

    def pretrain_decoder_run(self):
        self._model.disable_grad_all()
        self._model.enable_grad(from_="Up5" if self._disable_grad_encoder else "Conv1", util=self._extract_position)
        self.to(self._device)

        for self._cur_epoch in range(self._start_epoch, self._max_epoch_train_decoder):
            # todo: 1. to improve the foldersplit function
            #       2. to improve the gradient flow management for different representation

            pretrain_decoder_dict = PretrainDecoderEpoch(
                model=self._model, projection_head=self._projector,
                optimizer=self._optimizer,
                pretrain_decoder_loader=self._pretrain_loader_iter,
                contrastive_criterion=SupConLoss(), num_batches=self._num_batches,
                cur_epoch=self._cur_epoch, device=self._device, feature_extractor=self._feature_extractor
            ).run()
            self._scheduler.step()
            storage_dict = StorageIncomeDict(PRETRAIN_DECODER=pretrain_decoder_dict, )
            self._pretrain_encoder_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_dict, self._cur_epoch)
            self._save_to("last.pth", path=os.path.join(self._save_dir, "pretrain_decoder"))
        self.train_decoder_done = True

    def finetune_network_init(self, lr: float = 1e-7, weight_decay: float = 1e-5, multiplier: int = 200, warmup_max=10):

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer,
                                                                     self._max_epoch_train_finetune - warmup_max, 0)
        self._scheduler = GradualWarmupScheduler(self._optimizer, multiplier, warmup_max, self._scheduler)
        self._sup_criterion = KL_div()

        # set augmentation method as `total_freedom = True`
        assert hasattr(self._fine_tune_loader.dataset._transform, "_total_freedom")
        self._fine_tune_loader.dataset._transform._total_freedom = True  # noqa
        self._fine_tune_loader_iter = iter(self._fine_tune_loader)  # noqa

    def finetune_network_run(self, epocher_type=SimpleFineTuneEpoch):
        self.to(self._device)
        self._model.enable_grad_decoder()  # noqa
        self._model.enable_grad_encoder()  # noqa

        for self._cur_epoch in range(self._start_epoch, self._max_epoch_train_finetune):
            finetune_dict = epocher_type.create_from_trainer(self).run()
            val_dict, cur_score = EvalEpoch(self._model, val_loader=self._val_loader, sup_criterion=self._sup_criterion,
                                            cur_epoch=self._cur_epoch, device=self._device).run()
            self._scheduler.step()
            storage_dict = StorageIncomeDict(finetune=finetune_dict, val=val_dict)
            self._finetune_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_dict, self._cur_epoch)
            self.save(cur_score, os.path.join(self._save_dir, "finetune"))

    def start_training(
        self, checkpoint: str = None,
        pretrain_encoder_init_options=None,
        pretrain_decoder_init_options=None,
        finetune_network_init_options=None
    ):

        if finetune_network_init_options is None:
            finetune_network_init_options = {}
        if pretrain_decoder_init_options is None:
            pretrain_decoder_init_options = {}
        if pretrain_encoder_init_options is None:
            pretrain_encoder_init_options = {}
        with SummaryWriter(str(self._save_dir)) as self._writer:  # noqa
            if self.train_encoder:
                self.pretrain_encoder_init(**pretrain_encoder_init_options)
                if checkpoint is not None:
                    try:
                        self.load_state_dict_from_path(os.path.join(checkpoint, "pretrain_encoder"))
                    except Exception as e:
                        raise RuntimeError(f"loading pretrain_encoder_checkpoint failed with {e}, ")

                if not self.train_encoder_done:
                    self.pretrain_encoder_run()
            if self.train_decoder:
                self.pretrain_decoder_init(**pretrain_decoder_init_options)
                if checkpoint is not None:
                    try:
                        self.load_state_dict_from_path(os.path.join(checkpoint, "pretrain_decoder"))
                    except Exception as e:
                        print(f"loading pretrain_decoder_checkpoint failed with {e}, ")
                if not self.train_decoder_done:
                    self.pretrain_decoder_run()
            self.finetune_network_init(**finetune_network_init_options)
            if checkpoint is not None:
                try:
                    self.load_state_dict_from_path(os.path.join(checkpoint, "finetune"))
                except Exception as e:
                    print(f"loading finetune_checkpoint failed with {e}, ")
            self.finetune_network_run()


class ContrastTrainerMT(ContrastTrainer):

    def finetune_network_init(self, lr: float = 1e-7, weight_decay: float = 1e-5, multiplier: int = 200, warmup_max=10,
                              transform_axis=[1, 2], reg_weight: float = 10.0):
        super().finetune_network_init(lr, weight_decay, multiplier, warmup_max)

        self._reg_weight = reg_weight
        self._reg_criterion = nn.MSELoss()
        self._transform_axis = transform_axis

        from contrastyou.arch import UNet
        from deepclustering2.models import ema_updater
        # here we initialize the MT
        self._teacher_model = UNet(**self._configuration["Arch"])
        for param in self._teacher_model.parameters():
            param.detach_()
        self._teacher_model.train()
        self._ema_updater = ema_updater(alpha=0.999, justify_alpha=True, weight_decay=1e-6, update_bn=False)

    def finetune_network_run(self, epocher_type=MeanTeacherEpocher):
        self.to(self._device)
        self._model.enable_grad_decoder()  # noqa
        self._model.enable_grad_encoder()  # noqa

        for self._cur_epoch in range(self._start_epoch, self._max_epoch_train_finetune):
            finetune_dict = epocher_type.create_from_trainer(self).run()
            val_dict, cur_score = EvalEpoch(self._teacher_model, val_loader=self._val_loader,
                                            sup_criterion=self._sup_criterion,
                                            cur_epoch=self._cur_epoch, device=self._device).run()
            self._scheduler.step()
            storage_dict = StorageIncomeDict(finetune=finetune_dict, val=val_dict)
            self._finetune_storage.put_from_dict(storage_dict, epoch=self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_dict, self._cur_epoch)
            self.save(cur_score, os.path.join(self._save_dir, "finetune"))
