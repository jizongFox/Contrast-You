from copy import deepcopy
from typing import Tuple

import torch

from deepclustering2.meters2 import EpochResultDict, StorageIncomeDict
from .trainer import SemiTrainer
from ..epochers import EvalEpocher


class UnderstandPSTrainer(SemiTrainer):
    def _start_training(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            train_result: EpochResultDict
            eval_result: EpochResultDict
            cur_score: float
            train_result = self.run_epoch()
            if self.on_master():
                with torch.no_grad():
                    eval_on_train, cur_score_on_train = self._eval_on_train_epoch()
                    eval_result, cur_score = self.eval_epoch()

            # update lr_scheduler
            if hasattr(self, "_scheduler"):
                self._scheduler.step()
            if self.on_master():
                storage_per_epoch = StorageIncomeDict(tra=train_result, val=eval_result, val_on_tra=eval_on_train)
                self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
                self._writer.add_scalar_with_StorageDict(storage_per_epoch, self._cur_epoch)
                # save_checkpoint
                self.save_on_score(cur_score)
                # save storage result on csv file.
                self._storage.to_csv(self._save_dir)

    def _eval_on_train_epoch(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        train_set = deepcopy(self._labeled_loader._dataset)
        val_set = self._val_loader.dataset
        train_set._transform = val_set.transform
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_set, batch_size=4, shuffle=False)
        evaler = EvalEpocher(self._model, val_loader=train_dataloader, sup_criterion=self._sup_criterion,
                             cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score
