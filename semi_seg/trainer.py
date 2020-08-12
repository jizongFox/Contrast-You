from pathlib import Path
from typing import Tuple

import torch
from deepclustering2.meters2 import EpochResultDict, StorageIncomeDict
from deepclustering2.schedulers import GradualWarmupScheduler
from deepclustering2.trainer import Trainer
from deepclustering2.trainer.trainer import T_loader, T_loss, T_optim
from torch import nn

from contrastyou import PROJECT_PATH
from contrastyou.losses.iic_loss import IIDSegmentationSmallPathLoss
from semi_seg._utils import LocalClusterWrappaer
from semi_seg.epocher import TrainEpocher, EvalEpocher, UDATrainEpocher, IICTrainEpocher, UDAIICEpocher

__all__ = ["trainer_zoos"]


class SemiTrainer(Trainer):
    RUN_PATH = str(Path(PROJECT_PATH) / "semi_seg" / "runs")

    feature_positions = ["Up_conv4", "Up_conv3"]

    def __init__(self, model: nn.Module, optimizer: T_optim, labeled_loader: T_loader, unlabeled_loader: T_loader,
                 val_loader: T_loader, sup_criterion: T_loss, save_dir: str = "base", max_epoch: int = 100,
                 num_batches: int = 100, device: str = "cpu", configuration=None):
        super().__init__(model, save_dir, max_epoch, num_batches, device, configuration)
        self._optimizer = optimizer
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion

        # create scheduler
        config = configuration.copy()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=config["Trainer"]["max_epoch"] - config["Scheduler"]["warmup_max"],
            eta_min=1e-7
        )
        scheduler = GradualWarmupScheduler(optimizer, config["Scheduler"]["multiplier"],
                                           total_epoch=config["Scheduler"]["warmup_max"],
                                           after_scheduler=scheduler)
        self._scheduler = scheduler

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = TrainEpocher(self._model, self._optimizer, self._labeled_loader, self._unlabeled_loader,
                               self._sup_criterion, 0, self._num_batches, self._cur_epoch, self._device,
                               feature_position=self.feature_positions)
        result = trainer.run()
        return result

    def _start_training(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            train_result: EpochResultDict
            eval_result: EpochResultDict
            cur_score: float
            train_result = self.run_epoch()
            with torch.no_grad():
                eval_result, cur_score = self.eval_epoch()
            # update lr_scheduler
            self._scheduler.step()
            storage_per_epoch = StorageIncomeDict(tra=train_result, val=eval_result)
            self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
            for k, v in storage_per_epoch.__dict__.items():
                self._writer.add_scalar_with_tag(k, v, global_step=self._cur_epoch)
            # save_checkpoint
            self.save(cur_score)
            # save storage result on csv file.
            self._storage.to_csv(self._save_dir)

    def _eval_epoch(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        evaler = EvalEpocher(self._model, val_loader=self._val_loader, sup_criterion=self._sup_criterion,
                             cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score

    @classmethod
    def set_feature_positions(cls, feature_positions):
        cls.feature_positions = feature_positions


class UDATrainer(SemiTrainer):
    def __init__(self, model: nn.Module, optimizer: T_optim, labeled_loader: T_loader, unlabeled_loader: T_loader,
                 val_loader: T_loader, sup_criterion: T_loss, save_dir: str = "base", max_epoch: int = 100,
                 num_batches: int = 100, device: str = "cpu", configuration=None, reg_weight=1.0):
        super().__init__(model, optimizer, labeled_loader, unlabeled_loader, val_loader, sup_criterion, save_dir,
                         max_epoch, num_batches, device, configuration)
        self._reg_criterion = nn.MSELoss()
        self._reg_weight = reg_weight

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = UDATrainEpocher(self._model, self._optimizer, self._labeled_loader, self._unlabeled_loader,
                                  self._sup_criterion, reg_criterion=self._reg_criterion, reg_weight=self._reg_weight,
                                  num_batches=self._num_batches, cur_epoch=self._cur_epoch, device=self._device,
                                  feature_position=self.feature_positions)
        result = trainer.run()
        return result


class IICTrainer(SemiTrainer):
    def __init__(self, model: nn.Module, projector_wrappers: LocalClusterWrappaer, optimizer: T_optim,
                 labeled_loader: T_loader, unlabeled_loader: T_loader, val_loader: T_loader, sup_criterion: T_loss,
                 save_dir: str = "base", max_epoch: int = 100, num_batches: int = 100, reg_weight=0.0,
                 device: str = "cpu", configuration=None, IIDSegCriterion: IIDSegmentationSmallPathLoss = None
                 ):
        super().__init__(model, optimizer, labeled_loader, unlabeled_loader, val_loader, sup_criterion, save_dir,
                         max_epoch, num_batches, device, configuration)
        assert IIDSegCriterion is not None, IIDSegCriterion
        self._IIDSegCriterion = IIDSegCriterion
        self._projector_wrappers = projector_wrappers
        self._reg_weight = reg_weight

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = IICTrainEpocher(self._model, self._projector_wrappers, self._optimizer, self._labeled_loader,
                                  self._unlabeled_loader,
                                  self._sup_criterion, IIDSegCriterion=self._IIDSegCriterion,
                                  reg_weight=self._reg_weight,
                                  num_batches=self._num_batches, cur_epoch=self._cur_epoch, device=self._device,
                                  feature_position=self.feature_positions)
        result = trainer.run()
        return result


class UDAIICTrainer(IICTrainer):

    def __init__(self, model: nn.Module, optimizer: T_optim, labeled_loader: T_loader, unlabeled_loader: T_loader,
                 val_loader: T_loader, sup_criterion: T_loss, save_dir: str = "base", max_epoch: int = 100,
                 num_batches: int = 100, device: str = "cpu", configuration=None, cluster_parameters={},
                 IIDSeg_parameters={}, cons_weight=1, iic_weight=0.1):
        super().__init__(model, optimizer, labeled_loader, unlabeled_loader, val_loader, sup_criterion, save_dir,
                         max_epoch, num_batches, device, configuration, cluster_parameters, IIDSeg_parameters)
        self._cons_weight = cons_weight
        self._iic_weight = iic_weight

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = UDAIICEpocher(self._model, self._projector_wrappers, self._optimizer, self._labeled_loader,
                                self._unlabeled_loader,
                                self._sup_criterion, IIDSegCriterion=self._IIDSegCriterion,
                                reg_weight=1.0, reg_criterion=nn.MSELoss(),
                                num_batches=self._num_batches, cur_epoch=self._cur_epoch, device=self._device,
                                feature_position=self.feature_positions, cons_weight=self._cons_weight,
                                iic_weight=self._iic_weight)
        result = trainer.run()
        return result


trainer_zoos = {
    "partial": SemiTrainer,
    "uda": UDATrainer,
    "iic": IICTrainer,
    "udaiic": UDAIICTrainer
}
