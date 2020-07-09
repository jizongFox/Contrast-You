from pathlib import Path
from typing import Union, Callable

import torch
from deepclustering import ModelMode
from deepclustering.dataset import PatientSampler
from deepclustering.model import Model
from deepclustering.trainer._trainer import _Trainer
from deepclustering.utils import simplex, class2one_hot
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from tqdm import tqdm

from contrastyou import PROJECT_PATH, DATA_PATH
from contrastyou.dataloader._seg_datset import ContrastBatchSampler
from contrastyou.dataloader.acdc_dataset import ACDCSemiInterface
from contrastyou.meters2 import AverageValueMeter, UniversalDice


class Trainer(_Trainer):
    RUN_PATH = Path(PROJECT_PATH) / "runs"

    def __init__(self, model: Model, labeled_loader: Union[DataLoader, _BaseDataLoaderIter],
                 unlabeled_loader: Union[DataLoader, _BaseDataLoaderIter], val_loader: DataLoader,
                 sup_criterion=Callable[[Tensor, Tensor], Tensor],
                 reg_criterion=Callable[[Tensor, Tensor], Tensor],
                 max_epoch: int = 100, save_dir: str = "base", device="cpu",
                 num_batches: int = None,
                 config: dict = None, *args, **kwargs) -> None:
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        super().__init__(model, None, val_loader, max_epoch, save_dir, None, device, config, *args,
                         **kwargs)
        self._sup_criterion = sup_criterion
        self._reg_criterion = reg_criterion
        self._num_batches = num_batches

    def register_meters(self, enable_drawer=True) -> None:
        super(Trainer, self).register_meters(enable_drawer)
        self._meter_interface.register_new_meter("tra_loss", AverageValueMeter(), group_name="tra")
        self._meter_interface.register_new_meter("tra_dice", UniversalDice(C=4), group_name="tra")
        self._meter_interface.register_new_meter("reg_loss", AverageValueMeter(), group_name="tra")
        self._meter_interface.register_new_meter("val_loss", AverageValueMeter(), group_name="val")
        self._meter_interface.register_new_meter("val_dice", UniversalDice(C=4), group_name="val")

    def _train_loop(
        self,
        epoch: int = 0,
        mode=ModelMode.TRAIN,
        *args,
        **kwargs,
    ):
        self._model.set_mode(mode)
        assert self._model.training, self._model.training
        labeled_loader = self._labeled_loader
        unlabeled_loader = self._unlabeled_loader
        indicator = tqdm(zip(range(self._num_batches), labeled_loader, unlabeled_loader))
        for batch_num, label_data, unlabel_data in indicator:
            limage, ltarget, lfilename, partition_list, group_list = \
                label_data[0][0].to(self._device), label_data[0][1].to(self._device), label_data[1], label_data[2], \
                label_data[3]
            uimage = unlabel_data[0][0].to(self._device)
            predict_logits = self._model(torch.cat([limage, uimage], dim=0), force_simplex=False)
            assert not simplex(predict_logits), predict_logits
            label_logits, unlabel_logits = predict_logits[:len(limage)], predict_logits[len(limage):]
            onehot_ltarget = class2one_hot(ltarget.squeeze(), 4)

            sup_loss = self._sup_criterion(label_logits.softmax(1), onehot_ltarget)
            reg_loss = self._reg_criterion(unlabel_logits.softmax(1))

            total_loss = sup_loss + reg_loss * 0.0001

            self._model.zero_grad()
            total_loss.backward()
            self._model.step()
            with torch.no_grad():
                self._meter_interface["tra_loss"].add(sup_loss.item())
                self._meter_interface["tra_dice"].add(label_logits.max(1)[1], ltarget.squeeze(1),group_name = list(group_list))
                self._meter_interface["reg_loss"].add(reg_loss.item())
                report_dict = self._meter_interface.tracking_status("tra")
                indicator.set_postfix(report_dict)

    def _eval_loop(self, val_loader: Union[DataLoader, _BaseDataLoaderIter] = None, epoch: int = 0, mode=ModelMode.EVAL,
                   *args, **kwargs) -> float:
        assert isinstance(val_loader, DataLoader), type(val_loader)
        self._model.set_mode(mode)
        assert not self._model.training
        for batch_num, val_data in enumerate(val_loader):
            vimage, vtarget, vfilename = val_data[0][0].to(self._device), val_data[0][1].to(self._device), \
                                         val_data[1]
            predict_logits = self._model(vimage, force_simplex=False)
            onehot_target = class2one_hot(vtarget.squeeze(1), 4, )
            val_loss = self._sup_criterion(predict_logits.softmax(1), onehot_target, disable_assert=True)
            self._meter_interface["val_loss"].add(val_loss.item())
            self._meter_interface["val_dice"].add(predict_logits.max(1)[1], vtarget.squeeze(1))
        return self._meter_interface["val_dice"].summary()["DSC_mean"]

    def _start_training(self):
        for epoch in range(self._start_epoch, self._max_epoch):
            if self._model.get_lr() is not None:
                self._meter_interface["lr"].add(self._model.get_lr()[0])
            self.train_loop(epoch=epoch, )
            with torch.no_grad():
                current_score = self.eval_loop(self._val_loader, epoch)
            self._model.schedulerStep()
            # save meters and checkpoints
            self._meter_interface.step()
            if hasattr(self, "_dataframe_drawer"):
                self._dataframe_drawer()
            self.save_checkpoint(self.state_dict(), epoch, current_score)
            self._meter_interface.summary().to_csv(self._save_dir / "wholeMeter.csv")


acdc_manager = ACDCSemiInterface(root_dir=DATA_PATH, labeled_data_ratio=0.1, unlabeled_data_ratio=0.9)
label_set, unlabel_set, val_set = acdc_manager._create_semi_supervised_datasets()

labeled_loader = DataLoader(label_set,
                            batch_sampler=ContrastBatchSampler(label_set, group_sample_num=4, partition_sample_num=1),
                            num_workers=4)
unlabeled_loader = DataLoader(unlabel_set, batch_sampler=ContrastBatchSampler(unlabel_set, group_sample_num=4,
                                                                              partition_sample_num=1), num_workers=4)
val_loader = DataLoader(val_set,
                        batch_sampler=PatientSampler(
                            val_set,
                            grp_regex=val_set.dataset_pattern,
                            shuffle=False)
                        )

from deepclustering.arch import Enet
from deepclustering.loss import KL_div

arch = Enet(num_classes=4)
optim = torch.optim.Adam(arch.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, 0.1)
model = Model(arch, optim, scheduler)
from deepclustering.loss import Entropy
trainer = Trainer(model, labeled_loader, unlabeled_loader, val_loader, sup_criterion=KL_div(),
                  reg_criterion=Entropy(), device="cuda", num_batches=20)

trainer.start_training()
