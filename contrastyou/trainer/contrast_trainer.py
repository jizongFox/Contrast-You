from pathlib import Path
from typing import Union

from torch.utils.data import DataLoader

from contrastyou import PROJECT_PATH
from deepclustering2.meters2 import EpochResultDict
from deepclustering2.models import Model
from deepclustering2.trainer.trainer import Trainer, T_loader, T_loss


class ContrastTrainer(Trainer):
    RUN_PATH = Path(PROJECT_PATH) / "runs"

    def __init__(self, model: Model, tra_loader: T_loader, labeled_loader: T_loader, unlabeled_loader: T_loader,
                 fine_tune_loader: T_loader, val_loader: DataLoader, sup_criterion: T_loss, contrast_criterion=T_loss,
                 save_dir: str = "base", max_epoch_train_encoder: int = 100, max_epoch_train_decoder: int = 100,
                 max_epoch_train_finetune: int = 100,
                 num_batches: int = 256, reg_weight=0.0001, device: str = "cpu", configuration=None):
        super().__init__(model, save_dir, None, num_batches, device, configuration)  # noqa
        self._tra_loader = tra_loader
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._fine_tune_loader = fine_tune_loader
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion
        self._contrast_criterion = contrast_criterion
        self._reg_weight = reg_weight

        self._max_epoch_train_encoder = max_epoch_train_encoder
        self._max_epoch_train_decoder = max_epoch_train_decoder
        self._max_epoch_train_finetune = max_epoch_train_finetune

    def _run_train_encoder(self) -> EpochResultDict:
        pass

    def _run_train_decoder(self) -> EpochResultDict:
        pass

    def _run_finetuning(self) -> Union[EpochResultDict, float]:
        pass

    def _train_encoder(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            pass

    def _train_decoder(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            pass

    def _start_finetuning(self):
        pass

    def start_running(self):
        pass
