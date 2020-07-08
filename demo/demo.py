# demo for the framework
from typing import Union, Callable, Optional

import torch
from deepclustering.loss import KL_div
from haxiolearn.utils import class2one_hot
from torch import Tensor
from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter
from torchvision.transforms import ToTensor

from contrastyou import ModelState
from contrastyou.epoch._epoch import Epoch, REPORT_DICT
from contrastyou.meters2 import AverageValueMeter, ConfusionMatrix, MeterInterface
from contrastyou.modules.model import Model
from contrastyou.trainer._trainer import Trainer, EpochResult


# define epocher
class TrainEpoch(Epoch):

    def __init__(self, model: Model, loader: Union[DataLoader, _BaseDataLoaderIter], num_batches: Optional[int],
                 criterion: Callable[[Tensor, Tensor], Tensor], device="cpu", cur_epoch: int = 0) -> None:
        super().__init__(model=model, device=device, cur_epoch=cur_epoch)
        assert isinstance(loader, (DataLoader, _BaseDataLoaderIter)), type(loader)
        self._loader = loader
        self._num_batches = num_batches
        self._criterion = criterion

    @classmethod
    def create_from_trainer(cls, trainer, loader=None):
        model = trainer._model
        device = trainer._device
        cur_epoch = trainer._cur_epoch
        criterion = trainer._criterion
        num_batches = trainer._num_batches
        return cls(model=model, cur_epoch=cur_epoch, device=device, num_batches=num_batches,
                   criterion=criterion, loader=loader)

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter("loss", AverageValueMeter())
        meters.register_meter("confusion_mx", ConfusionMatrix(10))
        return meters

    def _run(self, mode=ModelState.TRAIN):
        self._model.set_mode(mode)
        self.assert_model_status()
        if "lr" in self.meters.meters:
            self.meters["lr"].add(self._model.get_lr()[0])
        epoch_iterator = enumerate(self._loader)
        if isinstance(self._num_batches, int) and self._num_batches > 1:
            epoch_iterator = zip(range(self._num_batches), self._loader)
        # main loop
        report_dict = None
        for self._cur_batch, data in epoch_iterator:
            report_dict = self.step(data)
        return report_dict

    def _step(self, data) -> REPORT_DICT:
        image, target = self._prepare_batch(data, self._device)
        onehot_target = class2one_hot(target, 10)
        predict_simplex = self._model(image, force_simplex=True)
        loss = self._criterion(predict_simplex, onehot_target)
        self._model.zero_grad()
        loss.backward()
        self._model.step()
        self.meters["loss"].add(loss.item())
        self.meters["confusion_mx"].add(predict_simplex.max(1)[1], target)
        report_dict = self.meters.tracking_status()
        return report_dict

    def _prepare_batch(self, data, device="cpu"):
        return data[0].to(device), data[1].to(device)


class ValEpoch(TrainEpoch):

    def __init__(self, model: Model, loader: DataLoader, criterion: Callable[[Tensor, Tensor], Tensor], device="cpu",
                 cur_epoch=0
                 ) -> None:
        assert isinstance(loader, DataLoader), f"{self.__class__.__name__} requires DataLoader type loader, " \
                                               f"given {loader.__class__.__name__}."
        super().__init__(model, loader, None, criterion, device, cur_epoch=cur_epoch)

    @classmethod
    def create_from_trainer(cls, trainer, loader=None):
        model = trainer._model
        device = trainer._device
        cur_epoch = trainer._cur_epoch
        criterion = trainer._criterion
        return cls(model=model, cur_epoch=cur_epoch, device=device,
                   criterion=criterion, loader=loader)

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(ValEpoch, self)._configure_meters(meters)
        meters.delete_meter("lr")
        return meters

    def _step(self, data) -> REPORT_DICT:
        image, target = self._prepare_batch(data, self._device)
        onehot_target = class2one_hot(target, 10)
        predict_simplex = self._model(image, force_simplex=True)
        loss = self._criterion(predict_simplex, onehot_target, disable_assert=True)
        self.meters["loss"].add(loss.item())
        self.meters["confusion_mx"].add(predict_simplex.max(1)[1], target)
        report_dict = self.meters.tracking_status()
        return report_dict

    def assert_model_status(self):
        assert not self._model.training, self._model.training


class Trainer(Trainer):

    def __init__(self, model: Model, train_loader: Union[DataLoader, _BaseDataLoaderIter], val_loader: DataLoader,
                 max_epoch: int = 100, save_dir: str = "base", device="cpu", num_batches=None,
                 criterion: Callable[[Tensor, Tensor], Tensor] = None,
                 config: dict = None) -> None:
        super().__init__(model, train_loader, val_loader, max_epoch, save_dir, device, config)
        self._criterion = criterion
        self._num_batches = num_batches

    def _start_training(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            epoch_result = self.start_single_epoch()
        return self._storage.summary()

    def _start_single_epoch(self):
        tra_epoch = TrainEpoch.create_from_trainer(self, self._train_loader)
        tra_epoch.register_callbacks(self._epoch_callbacks._train_callbacks)
        tra_result = tra_epoch.run(mode=ModelState.TRAIN)
        with torch.no_grad():
            val_epoch = ValEpoch.create_from_trainer(self, self._val_loader)
            val_epoch.register_callbacks(self._epoch_callbacks._val_callbacks)
            val_result = val_epoch.run(mode=ModelState.TEST)
        return EpochResult(train_result=tra_result, val_result=val_result)


if __name__ == '__main__':
    from torchvision.models import resnet18
    from torchvision.datasets import CIFAR10
    from contrastyou.callbacks import TQDMCallback, PrintResultCallback, SummaryCallback, SchedulerCallback, \
        EpochCallBacks, StorageCallback

    arch = resnet18(num_classes=10)
    optim = torch.optim.Adam(arch.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, 0.1)
    model = Model(arch, optim, scheduler)
    dataset = CIFAR10(root="./", transform=ToTensor(), download=True)
    val_dataset = CIFAR10(root="./", transform=ToTensor(), download=True, train=False)
    dataloader = DataLoader(dataset, batch_size=100, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, num_workers=4)
    trainer = Trainer(model, dataloader, val_loader, max_epoch=1, num_batches=20, save_dir="123", device="cuda",
                      criterion=KL_div())
    # trainer callback
    scheduler_cb = SchedulerCallback()
    writer = SummaryCallback(str(trainer._save_dir))
    storage_cb = StorageCallback()
    printable_callback = PrintResultCallback()
    trainer.register_callbacks([scheduler_cb, writer, storage_cb, printable_callback])
    # epoch callback
    printable_callback = PrintResultCallback()
    tqdm_indicator = TQDMCallback(frequency_print=10)
    trainer.register_epoch_callbacks(
        EpochCallBacks([printable_callback, tqdm_indicator],
                       [printable_callback, tqdm_indicator]))
    trainer.start_training()
