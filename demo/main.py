from pathlib import Path

import torch
from torch.utils.data import DataLoader

from contrastyou import PROJECT_PATH, DATA_PATH
from contrastyou.augment import ACDC_transforms
from contrastyou.dataloader._seg_datset import ContrastBatchSampler
from contrastyou.dataloader.acdc_dataset import ACDCSemiInterface
from deepclustering2 import ModelMode
from deepclustering2.arch import Enet, Tuple
from deepclustering2.dataset import PatientSampler
from deepclustering2.epoch._epocher import _Epocher
from deepclustering2.loss import KL_div, simplex
from deepclustering2.meters2 import EpochResultDict, MeterInterface, AverageValueMeter, UniversalDice
from deepclustering2.models import Model
from deepclustering2.tqdm import tqdm
from deepclustering2.trainer.trainer import _Trainer, T_loader, T_loss
from deepclustering2.type import to_device
from deepclustering2.utils import class2one_hot, set_benchmark

set_benchmark(1)


class TrainEpoch(_Epocher):

    def __init__(self, model: Model, labeled_loader: T_loader, unlabeled_loader: T_loader, sup_criteiron: T_loss,
                 reg_criterion: T_loss, num_batches: int = 100, cur_epoch=0, device="cpu",
                 reg_weight: float = 0.001) -> None:
        super().__init__(model, cur_epoch, device)
        assert isinstance(num_batches, int) and num_batches > 0, num_batches
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._sup_criterion = sup_criteiron
        self._reg_criterion = reg_criterion
        self._num_batches = num_batches
        self._reg_weight = reg_weight

    @classmethod
    def create_from_trainer(cls, trainer):
        return cls(trainer._model, trainer._labeled_loader, trainer._unlabeled_loader, trainer._sup_criterion,
                   trainer._reg_criterion,
                   trainer._num_batches, trainer._cur_epoch, trainer._device, trainer._reg_weight)

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("reg_loss", AverageValueMeter())
        meters.register_meter("ds", UniversalDice(4, [1, 2, 3]))
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.set_mode(ModelMode.TRAIN)
        assert self._model.training, self._model.training
        report_dict: EpochResultDict
        self.meters["lr"].add(self._model.get_lr()[0])

        with tqdm(range(self._num_batches)).set_description(desc=f"Training {self._cur_epoch}") as indicator:
            for i, label_data, unlabel_data in zip(indicator, self._labeled_loader, self._unlabeled_loader):
                [(labelimage, labeltarget), (labelimage_tf, labeltarget_tf)], filename, partition_list, group_list = \
                    to_device(label_data[0], self._device), label_data[1], label_data[2], label_data[3]
                unlabelimage, unlabelimage_tf = to_device([unlabel_data[0][0][0], unlabel_data[0][1][0]], self._device)

                predict_logits = self._model(
                    torch.cat([labelimage, labelimage_tf, unlabelimage, unlabelimage_tf], dim=0),
                    force_simplex=False)
                assert not simplex(predict_logits), predict_logits
                label_logit, label_logit_tf, unlabel_logit, unlabel_logit_tf \
                    = torch.split(predict_logits,
                                  [len(labelimage), len(labelimage_tf), len(unlabelimage), len(unlabelimage_tf)],
                                  dim=0)
                onehot_ltarget = class2one_hot(torch.cat([labeltarget.squeeze(), labeltarget_tf.squeeze()], dim=0), 4)
                sup_loss = self._sup_criterion(torch.cat([label_logit, label_logit_tf], dim=0).softmax(1),
                                               onehot_ltarget)
                reg_loss = self._reg_criterion(unlabel_logit.softmax(1), unlabel_logit_tf.softmax(1))
                total_loss = sup_loss + reg_loss * self._reg_weight

                self._model.zero_grad()
                total_loss.backward()
                self._model.step()

                with torch.no_grad():
                    self.meters["sup_loss"].add(sup_loss.item())
                    self.meters["ds"].add(label_logit.max(1)[1], labeltarget.squeeze(1),
                                          group_name=list(group_list))
                    self.meters["reg_loss"].add(reg_loss.item())
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
            report_dict = self.meters.tracking_status()
        return report_dict


class EvalEpoch(TrainEpoch):

    def __init__(self, model: Model, val_loader: DataLoader, sup_criteiron: T_loss,
                 cur_epoch=0, device="cpu") -> None:
        super().__init__(model, None, None, sup_criteiron, None, 1000, cur_epoch,
                         device, 0)
        self._val_loader = val_loader
        assert isinstance(val_loader, DataLoader), val_loader

    @classmethod
    def create_from_trainer(cls, trainer):
        return cls(trainer._model, trainer._val_loader, trainer._sup_criterion, trainer._cur_epoch, trainer._device)

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.delete_meters(["lr", "reg_loss"])
        return meters

    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        self._model.set_mode(ModelMode.EVAL)
        assert not self._model.training, self._model.training
        report_dict: EpochResultDict

        with tqdm(self._val_loader).set_description(f"Val Epoch {self._cur_epoch}") as indicator:
            for i, val_data in enumerate(indicator):
                vimage, vtarget, vfilename = val_data[0][0].to(self._device), val_data[0][1].to(self._device), \
                                             val_data[1]
                predict_logits = self._model(vimage, force_simplex=False)
                onehot_target = class2one_hot(vtarget.squeeze(1), 4, )
                val_loss = self._sup_criterion(predict_logits.softmax(1), onehot_target, disable_assert=True)
                with torch.no_grad():
                    self.meters["sup_loss"].add(val_loss.item())
                    self.meters["ds"].add(predict_logits.max(1)[1], vtarget.squeeze(1))
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
            report_dict = self.meters.tracking_status()
        return report_dict, self.meters["ds"].summary()["DSC_mean"]


class Trainer(_Trainer):
    RUN_PATH = Path(PROJECT_PATH) / "runs"

    def __init__(self, model: Model, labeled_loader: T_loader, unlabeled_loader: T_loader, val_loader: DataLoader,
                 sup_criterion: T_loss, reg_criterion=T_loss, save_dir: str = "base", max_epoch: int = 100,
                 num_batches: int = None, reg_weight=0.0001,
                 device: str = "cpu", configuration=None):
        super().__init__(model, save_dir, max_epoch, num_batches, device, configuration)

        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion
        self._reg_criterion = reg_criterion
        self._reg_weight = reg_weight

    def _run_epoch(self, epocher: _Epocher = TrainEpoch, *args, **kwargs) -> EpochResultDict:
        return super()._run_epoch(epocher, *args, **kwargs)

    def _eval_epoch(self, epocher: _Epocher = EvalEpoch, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        eval_epocher = epocher.create_from_trainer(trainer=self)
        return eval_epocher.run()


acdc_manager = ACDCSemiInterface(root_dir=DATA_PATH, labeled_data_ratio=0.1, unlabeled_data_ratio=0.9)

label_set, unlabel_set, val_set = acdc_manager._create_semi_supervised_datasets(
    labeled_transform=ACDC_transforms.train,
    unlabeled_transform=ACDC_transforms.train,
    val_transform=ACDC_transforms.val
)

labeled_loader = DataLoader(label_set,
                            batch_sampler=ContrastBatchSampler(label_set, group_sample_num=4, partition_sample_num=1),
                            num_workers=4
                            )
unlabeled_loader = DataLoader(unlabel_set,
                              batch_sampler=ContrastBatchSampler(unlabel_set, group_sample_num=4,
                                                                 partition_sample_num=1),
                              num_workers=4
                              )
val_loader = DataLoader(val_set,
                        batch_sampler=PatientSampler(
                            val_set,
                            grp_regex=val_set.dataset_pattern,
                            shuffle=False)
                        )
from contrastyou.losses.iic_loss import IIDSegmentationLoss

reg_criterion = IIDSegmentationLoss(padding=5)

arch = Enet(num_classes=4)
optim = torch.optim.Adam(arch.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 20, 0.1)
model = Model(arch, optim, scheduler)

trainer = Trainer(model, iter(labeled_loader), iter(unlabeled_loader), val_loader, sup_criterion=KL_div(),
                  reg_criterion=reg_criterion, device="cuda", num_batches=200)

# trainer.load_state_dict_from_path(trainer._save_dir / "last.pth")
# print(trainer.storage.summary())

trainer.start_training()
