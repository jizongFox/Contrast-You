from copy import deepcopy
from typing import Tuple, Type

import torch

from deepclustering2.meters2 import EpochResultDict, StorageIncomeDict
from semi_seg.epochers.base import EvalEpocher, TrainEpocher
from semi_seg.epochers.newepocher import NewEpocher, EvalEpocherWOEval
from semi_seg.trainers.trainer import SemiTrainer, InfoNCETrainer, IICTrainer
from .._utils import ContrastiveProjectorWrapper


class UnderstandPSTrainer(SemiTrainer):
    """
    This Trainer class is to understand the behavior of BN 
    """

    def _start_training(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            train_result: EpochResultDict
            eval_result: EpochResultDict
            cur_score: float
            train_result = self.run_epoch()
            if self.on_master():
                with torch.no_grad():
                    eval_on_labeled, _ = self._eval_on_labeled_epoch()
                    eval_on_unlabeled, _ = self._eval_on_unlabeled_epoch()
                    eval_result, cur_score = self.eval_epoch()

                    eval_train_on_labeled, _ = self._eval_on_labeled_epoch(eval=False)
                    eval_train_on_unlabeled, _ = self._eval_on_unlabeled_epoch(eval=False)
                    eval_train_result, _ = self.eval_epoch(eval=False)

            # update lr_scheduler
            if hasattr(self, "_scheduler"):
                self._scheduler.step()
            if self.on_master():
                storage_per_epoch = StorageIncomeDict(tra=train_result, val=eval_result, val_on_label=eval_on_labeled,
                                                      val_on_unlabeled=eval_on_unlabeled,
                                                      eval_train_on_labeled=eval_train_on_labeled,
                                                      eval_train_on_unlabeled=eval_train_on_unlabeled,
                                                      eval_train_result=eval_train_result)
                self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
                self._writer.add_scalar_with_StorageDict(storage_per_epoch, self._cur_epoch)
                # save_checkpoint
                self.save_on_score(cur_score)
                # save storage result on csv file.
                self._storage.to_csv(self._save_dir)

    def _eval_on_labeled_epoch(self, eval=True, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        train_set = deepcopy(self._labeled_loader._dataset)
        val_set = self._val_loader.dataset
        train_set._transform = val_set.transform
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_set, batch_size=4, shuffle=False)
        if eval:
            EvalClass = EvalEpocher
        else:
            EvalClass = EvalEpocherWOEval
        evaler = EvalClass(self._model, val_loader=train_dataloader, sup_criterion=self._sup_criterion,
                           cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score

    def _eval_on_unlabeled_epoch(self, eval=True, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        train_set = deepcopy(self._unlabeled_loader._dataset)
        val_set = self._val_loader.dataset
        train_set._transform = val_set.transform
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_set, batch_size=4, shuffle=False)
        if eval:
            EvalClass = EvalEpocher
        else:
            EvalClass = EvalEpocherWOEval
        evaler = EvalClass(self._model, val_loader=train_dataloader, sup_criterion=self._sup_criterion,
                           cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score

    def _eval_epoch(self, eval=True, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        if eval:
            EvalClass = EvalEpocher
        else:
            EvalClass = EvalEpocherWOEval
        evaler = EvalClass(self._model, val_loader=self._val_loader, sup_criterion=self._sup_criterion,
                           cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score


class ExperimentalTrainer(InfoNCETrainer):
    def _init(self):
        super(IICTrainer, self)._init()
        config = deepcopy(self._config["InfoNCEParameters"])
        self._projector = ContrastiveProjectorWrapper()
        self._projector.init_encoder(
            feature_names=self.feature_positions,
            **config["EncoderParams"]
        )
        self._projector.init_decoder(
            feature_names=self.feature_positions,
            **config["DecoderParams"]
        )
        from contrastyou.losses.contrast_loss import SupConLoss2 as SupConLoss

        self._criterion = SupConLoss(temperature=config["LossParams"]["temperature"], out_mode=True)
        self._reg_weight = float(config["weight"])

    def _set_epocher_class(self, epocher_class: Type[TrainEpocher] = NewEpocher):
        super(ExperimentalTrainer, self)._set_epocher_class(epocher_class)
