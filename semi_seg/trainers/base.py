import os
from pathlib import Path
from typing import Tuple, Type

import torch
from loguru import logger
from torch import nn
from torch import optim

from contrastyou import PROJECT_PATH
from deepclustering2 import optim
from deepclustering2.meters2 import EpochResultDict
from deepclustering2.meters2 import StorageIncomeDict
from deepclustering2.schedulers import GradualWarmupScheduler
from deepclustering2.tqdm import item2str
from deepclustering2.trainer2 import Trainer
from deepclustering2.type import T_loader, T_loss
from semi_seg.epochers import InferenceEpocher
from semi_seg.epochers import TrainEpocher, EvalEpocher, FineTuneEpocher


class SemiTrainer(Trainer):
    RUN_PATH = str(Path(PROJECT_PATH) / "semi_seg" / "runs")  # noqa

    def __init__(self, model: nn.Module, labeled_loader: T_loader, unlabeled_loader: T_loader,
                 val_loader: T_loader, sup_criterion: T_loss, save_dir: str = "base", max_epoch: int = 100,
                 num_batches: int = 100, device: str = "cpu", configuration=None, **kwargs):
        super().__init__(model, save_dir, max_epoch, num_batches, device, configuration)
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion

    # initialization
    def init(self):
        self._init()
        self._init_optimizer()
        self._init_scheduler(self._optimizer)

    def _init(self):
        self.set_feature_positions(self._config["Trainer"]["feature_names"])
        feature_importance = self._config["Trainer"]["feature_importance"]
        assert isinstance(feature_importance, list), type(feature_importance)
        feature_importance = [float(x) for x in feature_importance]
        self._feature_importance = [x / sum(feature_importance) for x in feature_importance]

        assert len(self._feature_importance) == len(self.feature_positions), \
            (self._feature_importance, self.feature_positions)

        logger.info("{} feature importance: {}", self.__class__.__name__,
                    item2str({c: v for c, v in zip(self.feature_positions, self._feature_importance)}))

        self._disable_bn = self._config["Trainer"].get("disable_bn_track_for_unlabeled_data", False)
        self._train_with_two_stage = self._config["Trainer"].get("two_stage_training", False)

    def _init_scheduler(self, optimizer):
        scheduler_dict = self._config.get("Scheduler", None)
        if scheduler_dict is None:
            return
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer,
                T_max=self._config["Trainer"]["max_epoch"] - self._config["Scheduler"]["warmup_max"],
                eta_min=1e-7
            )
            scheduler = GradualWarmupScheduler(optimizer, scheduler_dict["multiplier"],
                                               total_epoch=scheduler_dict["warmup_max"],
                                               after_scheduler=scheduler)
            self._scheduler = scheduler

    def _init_optimizer(self):
        optim_dict = self._config["Optim"]
        self._optimizer = optim.__dict__[optim_dict["name"]](
            params=self._model.parameters(),
            **{k: v for k, v in optim_dict.items() if k != "name"}
        )

    # run epoch
    def _set_epocher_class(self, epocher_class: Type[TrainEpocher] = TrainEpocher):
        self.epocher_class = epocher_class  # noqa

    def run_epoch(self, *args, **kwargs):
        self._set_epocher_class()
        epocher = self._run_init()
        return self._run_epoch(epocher, *args, **kwargs)

    def _run_init(self, ):
        epocher = self.epocher_class(
            model=self._model, optimizer=self._optimizer, labeled_loader=self._labeled_loader,
            unlabeled_loader=self._unlabeled_loader, sup_criterion=self._sup_criterion, num_batches=self._num_batches,
            cur_epoch=self._cur_epoch, device=self._device, feature_position=self.feature_positions,
            feature_importance=self._feature_importance, train_with_two_stage=self._train_with_two_stage,
            disable_bn_track_for_unlabeled_data=self._disable_bn
        )
        return epocher

    def _run_epoch(self, epocher: TrainEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=0.0, **kwargs)  # partial supervision without regularization
        result = epocher.run()
        return result

    # run eval
    def _eval_epoch(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        evaler = EvalEpocher(self._model, val_loader=self._val_loader, sup_criterion=self._sup_criterion,
                             cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score

    def _start_training(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            train_result: EpochResultDict
            eval_result: EpochResultDict
            cur_score: float
            train_result = self.run_epoch()
            if self.on_master():
                with torch.no_grad():
                    eval_result, cur_score = self.eval_epoch()
            # update lr_scheduler
            if hasattr(self, "_scheduler") and hasattr(self._scheduler, "step") and callable(self._scheduler.step):
                self._scheduler.step()
            if self.on_master():
                storage_per_epoch = StorageIncomeDict(tra=train_result, val=eval_result)
                self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
                self._writer.add_scalar_with_StorageDict(storage_per_epoch, self._cur_epoch)
                # save_checkpoint
                self.save_on_score(cur_score)
                # save storage result on csv file.
                self._storage.to_csv(self._save_dir)

    def inference(self, checkpoint=None):  # noqa
        """
        inference method is going to use InferenceEpocher which is with HD scores.
        :param checkpoint: str
        :return: inference result and criterion score
        """
        if checkpoint is None:
            self.load_state_dict_from_path(os.path.join(self._save_dir, "best.pth"), strict=True)
        else:
            checkpoint = Path(checkpoint)
            if checkpoint.is_file():
                if not checkpoint.suffix == ".pth":
                    raise FileNotFoundError(checkpoint)
            else:
                assert checkpoint.exists()
                checkpoint = checkpoint / "best.pth"
            self.load_state_dict_from_path(str(checkpoint), strict=True)
        evaler = InferenceEpocher(self._model, val_loader=self._val_loader, sup_criterion=self._sup_criterion,
                                  cur_epoch=self._cur_epoch, device=self._device)
        evaler.init(save_dir=self._save_dir)
        result, cur_score = evaler.run()
        return result, cur_score

    def set_feature_positions(self, feature_positions):
        logger.info("feature_position for {}: [{}]", self.__class__.__name__, ", ".join(feature_positions))
        self.feature_positions = feature_positions  # noqa


class FineTuneTrainer(SemiTrainer):

    def _set_epocher_class(self, epocher_class: Type[TrainEpocher] = FineTuneEpocher):
        super()._set_epocher_class(epocher_class)
