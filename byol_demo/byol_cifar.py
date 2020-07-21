# this is a demo to show how to inplement the BYOL for cifar dataset
import os
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from byol_demo.utils import loss_fn, TransTwice
from contrastyou import DATA_PATH, PROJECT_PATH
from deepclustering2 import ModelMode
from deepclustering2.augment import pil_augment
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.epoch import _Epocher
from deepclustering2.meters2 import EpochResultDict, StorageIncomeDict, Storage, MeterInterface, AverageValueMeter, \
    ConfusionMatrix
from deepclustering2.models import Model, EMA_Model
from deepclustering2.optim import get_lrs_from_optimizer, Optimizer
from deepclustering2.tqdm import tqdm
from deepclustering2.trainer.trainer import T_loader, Trainer
from deepclustering2.writer import SummaryWriter


# todo: redo that
class BYOLEpocher(_Epocher):
    def __init__(self, model: nn.Module, target_model: nn.Module, optimizer: Optimizer, data_loader: T_loader,
                 num_batches: int = 1000,
                 cur_epoch=0, device="cpu", ema_updater=None) -> None:
        super().__init__(model, cur_epoch, device)
        self._target_model = target_model
        self._optimizer = optimizer
        self._data_loader = data_loader
        assert isinstance(num_batches, int) and num_batches > 0, num_batches
        self._num_batches = num_batches
        self._l2_criterion = loss_fn
        self._ema_updater = ema_updater

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("contrastive_loss", AverageValueMeter())
        meters.register_meter("lr", AverageValueMeter())
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.set_mode(ModelMode.TRAIN)
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        assert self._model.training, self._model.training
        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
            for i, data in zip(indicator, self._data_loader):
                (img, img_tf), _ = self._preprocess_data(data, self._device)
                pred_img_project = self._model(img, return_prediction=True)
                with torch.no_grad():
                    pred_img_tf = self._target_model(img_tf, return_projection=True)
                loss = self._l2_criterion(pred_img_tf,
                                          pred_img_project).mean()
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._ema_updater(self._target_model, self._model)
                with torch.no_grad():
                    self.meters["contrastive_loss"].add(loss.item())
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
        return report_dict

    @staticmethod
    def _preprocess_data(data, device):
        return (data[0][0].to(device), data[0][1].to(device)), data[1].to(device)


# todo: redo that
class FineTuneEpocher(_Epocher):
    def __init__(self, model: Model, data_loader: T_loader, optimizer: Optimizer, num_batches: int = 1000,
                 cur_epoch=0, device="cpu") -> None:
        super().__init__(model, cur_epoch, device)
        self._data_loader = data_loader
        self._optimizer = optimizer
        self._sup_criterion = nn.CrossEntropyLoss()
        assert isinstance(num_batches, int) and num_batches > 0, num_batches
        self._num_batches = num_batches

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter("cmx", ConfusionMatrix(10))
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.set_mode(ModelMode.TRAIN)
        assert self._model.training, self._model.training
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
            for i, data in zip(indicator, self._data_loader):
                (img, img_tf), target = self._preprocess_data(data, self._device)
                with torch.no_grad():
                    pred_logits = self._model(img, return_classes=True)
                loss = self._sup_criterion(pred_logits, target)
                if torch.isnan(loss):
                    raise RuntimeError("nan in loss")
                with torch.no_grad():
                    self.meters["sup_loss"].add(loss.item())
                    self.meters["cmx"].add(pred_logits.max(1)[1], target)
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
        return report_dict

    @staticmethod
    def _preprocess_data(data, device):
        return (data[0][0].to(device), data[0][1].to(device)), data[1].to(device)


class EvalEpocher(FineTuneEpocher):

    def __init__(self, model: Model, val_loader, num_batches: int = 1000, cur_epoch=0,
                 device="cpu", *args, **kwargs) -> None:
        super().__init__(model, None, num_batches, cur_epoch, device)
        self._val_loader = val_loader

    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        self._model.eval()
        assert self._model.training, self._model.training
        with tqdm(self._val_loader).set_desc_from_epocher(self) as indicator:
            for i, data in enumerate(indicator):
                (img, img_tf), target = self._preprocess_data(data, self._device)
                pred_logits = self._model(img, return_classes=True)
                loss = self._sup_criterion(pred_logits, target)
                self.meters["sup_loss"].add(loss.item())
                self.meters["cmx"].add(pred_logits.max(1)[1], target)
                report_dict = self.meters.tracking_status()
                indicator.set_postfix_dict(report_dict)
        return report_dict, report_dict["cmx"]["acc"]


class BYOLTrainer(Trainer):
    RUN_PATH = str(Path(PROJECT_PATH) / "runs" / "byol")
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / "archives")

    def __init__(self, model: Model, pretrain_loader: T_loader, finetune_loader: T_loader, val_loader: T_loader,
                 save_dir: str = "base",
                 max_epoch_contrastive: int = 100, max_epoch_finetuning: int = 100, num_batches: int = 100,
                 device: str = "cpu", configuration=None):
        super().__init__(model, save_dir, 0, num_batches, device, configuration)

        assert self._max_epoch == 0, self._max_epoch
        self._pretrain_loader = pretrain_loader
        self._finetune_loader = finetune_loader
        self._max_epoch_contrastive = max_epoch_contrastive
        self._max_epoch_finetuning = max_epoch_finetuning
        self._register_buffer("_PRETRAIN_DONE", False)

        self._val_loader = val_loader
        self._pretrain_storage = Storage()
        self._finetune_storage = Storage()

    def contrastive_training_init(self):
        self._target_model = deepcopy(self._model)
        from deepclustering2.arch._init_utils import init_weights
        self._target_model.apply(init_weights)
        for param in self._target_model.parameters():
            param.detach_()

        self._target_model.train()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-6)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, self._max_epoch_contrastive - 10)
        from deepclustering2.schedulers import GradualWarmupScheduler
        self._scheduler = GradualWarmupScheduler(self._optimizer, 300, 10, self._scheduler)
        self._model._classhead.requires_grad = False
        from deepclustering2.models import ema_updater
        self._ema_updater = ema_updater(alpha=0.999, justify_alpha=False, weight_decay=0)

    def contrastive_training_run(self):
        save_path = os.path.join(self._save_dir, "pretrain")
        for self._cur_epoch in range(self._start_epoch, self._max_epoch_contrastive):
            pretrain_result: EpochResultDict
            pretrain_result = BYOLEpocher(self._model, self._target_model, self._optimizer, self._pretrain_loader,
                                          self._num_batches, self._cur_epoch,
                                          self._device, self._ema_updater)
            # update lr_scheduler
            self._scheduler.step()
            storage_per_epoch = StorageIncomeDict(pretrain=pretrain_result)
            self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_per_epoch, epoch=self._cur_epoch)
            # save_checkpoint
            if self._cur_epoch % 10 == 1:
                self.periodic_save(cur_epoch=self._cur_epoch, path=save_path)
            # save storage result on csv file.
            self._storage.to_csv(save_path)

    def funetuining_init(self):

    def fun(self):
        save_path = os.path.join(self._save_dir, "finetune")
        Path(save_path).mkdir(exist_ok=True, parents=True)
        with Storage() as self._storage:
            for self._cur_epoch in range(self._start_epoch, self._max_epoch_finetuning):
                finetune_result: EpochResultDict
                finetune_result = self.finetune_epoch()
                with torch.no_grad():
                    val_result, cur_score = self.eval_epoch()
                # update lr_scheduler
                self._classify_model.schedulerStep()
                storage_per_epoch = StorageIncomeDict(finetune=finetune_result, val=val_result)
                self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
                for k, v in storage_per_epoch.__dict__.items():
                    self._writer.add_scalar_with_tag(k, v, global_step=self._cur_epoch)
                # save_checkpoint
                # self.periodic_save(cur_epoch=self._cur_epoch, path=save_path)
                self.save(cur_score)
                # save storage result on csv file.
                self._storage.to_csv(save_path)

    def start_training(self):
        with SummaryWriter(str(self._save_dir)) as self._writer:
            if not self._PRETRAIN_DONE:
                self._start_contrastive_training()

            if not self._FINETUNE_DONE:
                self._start_funetuining()


if __name__ == '__main__':
    from contrastyou.arch.vgg import VGG11, ClassifyHead
    from randaugment import RandAugment
    from torchvision.datasets import CIFAR10
    from deepclustering2.schedulers import GradualWarmupScheduler

    lr = 1e-6
    net = VGG11()
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 90, 0)
    scheduler = GradualWarmupScheduler(optim, 300, 10, scheduler)
    model = Model(net, optim, scheduler)
    target_model = EMA_Model(deepcopy(model))

    classfy_model = ClassifyHead()
    class_optim = torch.optim.Adam(classfy_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(class_optim, 90, 0)
    scheduler = GradualWarmupScheduler(class_optim, 300, 10, scheduler)

    classfy_model = Model(classfy_model, class_optim, scheduler)

    rand_aug_transform = pil_augment.Compose([
        RandAugment(),
        pil_augment.ToTensor()
    ])
    finetune_transform = pil_augment.Compose([
        pil_augment.RandomCrop(size=32, padding=4),
        pil_augment.ToTensor()
    ])
    val_transform = pil_augment.Compose([
        pil_augment.ToTensor()
    ])
    pretra_data = CIFAR10(root=DATA_PATH, train=True, transform=TransTwice(rand_aug_transform), download=True)
    finetue_data = CIFAR10(root=DATA_PATH, train=True, transform=TransTwice(finetune_transform), download=True)
    val_data = CIFAR10(root=DATA_PATH, train=False, transform=TransTwice(val_transform), download=True)
    pretra_loader = DataLoader(pretra_data, sampler=InfiniteRandomSampler(pretra_data, shuffle=True), batch_size=32,
                               num_workers=8)
    finetune_loader = DataLoader(finetue_data, sampler=InfiniteRandomSampler(finetue_data, shuffle=True, ),
                                 batch_size=32, num_workers=8)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=50, num_workers=4)

    trainer = BYOLTrainer(model=model, target_model=target_model, classify_model=classfy_model, save_dir="byol",
                          device="cuda", pretrain_loader=iter(pretra_loader), finetune_loader=iter(finetune_loader),
                          val_loader=val_loader, max_epoch_contrastive=1, max_epoch_finetuning=100, num_batches=200)
    trainer.start_training()
