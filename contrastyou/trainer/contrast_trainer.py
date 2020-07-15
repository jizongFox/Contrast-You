from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader

from contrastyou import PROJECT_PATH
from contrastyou.epocher import contrast_epocher
from deepclustering2.epoch._epocher import _Epocher
from deepclustering2.meters2 import EpochResultDict
from deepclustering2.models import Model
from deepclustering2.trainer.trainer import _Trainer, T_loader, T_loss


class ContrastTrainer(_Trainer):
    RUN_PATH = Path(PROJECT_PATH) / "runs"

    def __init__(self, model: Model, tra_loader: T_loader, labeled_loader: T_loader, unlabeled_loader: T_loader,
                 val_loader: DataLoader,
                 sup_criterion: T_loss, reg_criterion=T_loss, save_dir: str = "base", max_epoch: int = 100,
                 num_batches: int = None, reg_weight=0.0001,
                 device: str = "cpu", configuration=None):
        super().__init__(model, save_dir, max_epoch, num_batches, device, configuration)
        self._tra_loader = tra_loader
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion
        self._reg_criterion = reg_criterion
        self._reg_weight = reg_weight

    def _run_epoch(self, epocher: _Epocher = contrast_epocher.TrainEpoch, *args, **kwargs) -> EpochResultDict:
        return super()._run_epoch(epocher, *args, **kwargs)

    def _train_encoder(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            pass

    def _train_decoder(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            pass

