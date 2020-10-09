import os
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Tuple, Type

import torch
from torch import nn

from contrastyou import PROJECT_PATH
from contrastyou.losses.iic_loss import IIDSegmentationSmallPathLoss
from deepclustering2 import optim
from deepclustering2.loss import KL_div
from deepclustering2.meters2 import EpochResultDict, StorageIncomeDict
from deepclustering2.models import ema_updater
from deepclustering2.schedulers import GradualWarmupScheduler
from deepclustering2.trainer2 import Trainer
from deepclustering2.type import T_loader, T_loss
from semi_seg._utils import IICLossWrapper, ProjectorWrapper
from semi_seg.epocher2 import TrainEpocher, EvalEpocher, UDATrainEpocher, IICTrainEpocher, UDAIICEpocher, \
    EntropyMinEpocher, MeanTeacherEpocher, IICMeanTeacherEpocher, InferenceEpocher, MIDLPaperEpocher

__all__ = ["trainer_zoos"]


class SemiTrainer(Trainer):
    RUN_PATH = str(Path(PROJECT_PATH) / "semi_seg" / "runs")  # noqa

    feature_positions = ["Up_conv4", "Up_conv3"]

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
        assert len(self._feature_importance) == len(self.feature_positions)

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
    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = TrainEpocher):
        self.epocher_class = epocher_class

    def run_epoch(self, *args, **kwargs):
        trainer = self._run_init()
        return self._run_epoch(trainer, *args, **kwargs)

    def _run_init(self, ):
        epocher = self.epocher_class(
            model=self._model, optimizer=self._optimizer, labeled_loader=self._labeled_loader,
            unlabeled_loader=self._unlabeled_loader, sup_criterion=self._sup_criterion, num_batches=self._num_batches,
            cur_epoch=self._cur_epoch, device=self._device, feature_position=self.feature_positions,
            feature_importance=self._feature_importance
        )
        return epocher

    def _run_epoch(self, epocher: TrainEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=0.0)  # partial supervision without regularization
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
            if hasattr(self, "_scheduler"):
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

    @classmethod
    def set_feature_positions(cls, feature_positions):
        cls.feature_positions = feature_positions


class UDATrainer(SemiTrainer):

    def _init(self):
        super(UDATrainer, self)._init()
        config = deepcopy(self._config["UDARegCriterion"])
        self._reg_criterion = {"mse": nn.MSELoss(), "kl": KL_div()}[config["name"]]
        self._reg_weight = float(config["weight"])

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = UDATrainEpocher):
        super().set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: UDATrainEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, reg_criterion=self._reg_criterion)
        result = epocher.run()
        return result


class MIDLTrainer(UDATrainer):
    def _init(self):
        super(MIDLTrainer, self)._init()
        self._uda_weight = deepcopy(self._reg_weight)
        self._reg_weight = 1.0
        config = deepcopy(self._config["MIDLPaperParameters"])
        self._iic_segcriterion = IIDSegmentationSmallPathLoss(
            padding=int(config["padding"]),
            patch_size=int(config["patch_size"])
        )
        self._iic_weight = float(config["iic_weight"])

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = MIDLPaperEpocher):
        super().set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: MIDLPaperEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(iic_weight=self._iic_weight, uda_weight=self._uda_weight, iic_segcriterion=self._iic_segcriterion,
                     reg_criterion=self._reg_criterion)
        result = epocher.run()
        return result


class IICTrainer(SemiTrainer):
    def _init(self):
        super(IICTrainer, self)._init()
        config = deepcopy(self._config["IICRegParameters"])
        self._projector_wrappers = ProjectorWrapper()
        self._projector_wrappers.init_encoder(
            feature_names=self.feature_positions,
            **config["EncoderParams"]
        )
        self._projector_wrappers.init_decoder(
            feature_names=self.feature_positions,
            **config["DecoderParams"]
        )
        self._IIDSegWrapper = IICLossWrapper(
            feature_names=self.feature_positions,
            **config["LossParams"]
        )
        self._reg_weight = float(config["weight"])

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = IICTrainEpocher):
        super().set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: IICTrainEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, projectors_wrapper=self._projector_wrappers,
                     IIDSegCriterionWrapper=self._IIDSegWrapper)
        result = epocher.run()
        return result

    def _init_optimizer(self):
        config = deepcopy(self._config["Optim"])
        self._optimizer = optim.__dict__[config["name"]](
            params=chain(self._model.parameters(), self._projector_wrappers.parameters()),
            **{k: v for k, v in config.items() if k != "name"}
        )


class UDAIICTrainer(IICTrainer):

    def _init(self):
        super(UDAIICTrainer, self)._init()
        self._iic_weight = deepcopy(self._reg_weight)
        UDA_config = deepcopy(self._config["UDARegCriterion"])
        self._reg_criterion = {"mse": nn.MSELoss(), "kl": KL_div()}[UDA_config["name"]]
        self._uda_weight = float(UDA_config["weight"])

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = UDAIICEpocher):
        super().set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: UDAIICEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(iic_weight=self._iic_weight, uda_weight=self._uda_weight,
                     projectors_wrapper=self._projector_wrappers, IIDSegCriterionWrapper=self._IIDSegWrapper,
                     reg_criterion=self._reg_criterion)
        result = epocher.run()
        return result


class EntropyMinTrainer(SemiTrainer):
    def _init(self):
        super(EntropyMinTrainer, self)._init()
        config = deepcopy(self._config["EntropyMinParameters"])
        self._reg_weight = float(config["weight"])

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = EntropyMinEpocher):
        super().set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: EntropyMinEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight)
        result = epocher.run()
        return result


class MeanTeacherTrainer(SemiTrainer):
    def _init(self):
        super(MeanTeacherTrainer, self)._init()
        self._teacher_model = deepcopy(self._model)
        for param in self._teacher_model.parameters():
            param.detach_()
        self._teacher_model.train()
        config = deepcopy(self._config["MeanTeacherParameters"])
        self._reg_criterion = {"mse": nn.MSELoss(), "kl": KL_div()}[config["name"]]
        self._ema_updater = ema_updater(alpha=float(config["alpha"]), weight_decay=float(config["weight_decay"]))
        self._reg_weight = float(config["weight"])

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = MeanTeacherEpocher):
        super().set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: MeanTeacherEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, teacher_model=self._teacher_model, reg_criterion=self._reg_criterion,
                     ema_updater=self._ema_updater)
        result = epocher.run()
        return result

    def _eval_epoch(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        evaler = EvalEpocher(self._teacher_model, val_loader=self._val_loader, sup_criterion=self._sup_criterion,
                             cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score


class IICMeanTeacherTrainer(IICTrainer):

    def _init(self):
        super()._init()
        self._iic_weight = deepcopy(self._reg_weight)
        self._teacher_model = deepcopy(self._model)
        for param in self._teacher_model.parameters():
            param.detach_()
        self._teacher_model.train()
        config = deepcopy(self._config["MeanTeacherParameters"])
        self._reg_criterion = {"mse": nn.MSELoss(), "kl": KL_div()}[config["name"]]
        self._ema_updater = ema_updater(alpha=float(config["alpha"]), weight_decay=float(config["weight_decay"]))
        self._mt_weight = float(config["weight"])

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = IICMeanTeacherEpocher):
        super().set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: IICMeanTeacherEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(projectors_wrapper=self._projector_wrappers, IIDSegCriterionWrapper=self._IIDSegWrapper,
                     reg_criterion=self._reg_criterion, teacher_model=self._teacher_model,
                     ema_updater=self._ema_updater, mt_weight=self._mt_weight, iic_weight=self._iic_weight)
        result = epocher.run()
        return result

    def _eval_epoch(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        evaler = EvalEpocher(self._teacher_model, val_loader=self._val_loader, sup_criterion=self._sup_criterion,
                             cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score


trainer_zoos = {
    "partial": SemiTrainer,
    "uda": UDATrainer,
    "iic": IICTrainer,
    "udaiic": UDAIICTrainer,
    "entropy": EntropyMinTrainer,
    "meanteacher": MeanTeacherTrainer,
    "iicmeanteacher": IICMeanTeacherTrainer,
    "midl": MIDLTrainer
}
