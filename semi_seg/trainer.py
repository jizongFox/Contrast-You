import os
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Tuple, Type

import torch
from torch import nn
from torch import optim

from contrastyou import PROJECT_PATH
from contrastyou.helper import get_dataset
from contrastyou.losses.contrast_loss import SupConLoss
from contrastyou.losses.iic_loss import IIDSegmentationSmallPathLoss
from deepclustering2 import optim
from deepclustering2.loss import KL_div
from deepclustering2.meters2 import EpochResultDict
from deepclustering2.meters2 import StorageIncomeDict
from deepclustering2.models import ema_updater
from deepclustering2.schedulers import GradualWarmupScheduler
from deepclustering2.schedulers.customized_scheduler import RampScheduler, WeightScheduler
from deepclustering2.trainer2 import Trainer
from deepclustering2.type import T_loader, T_loss
from semi_seg._utils import ClusterProjectorWrapper, IICLossWrapper, PICALossWrapper, ContrastiveProjectorWrapper
from semi_seg.epochers import IICTrainEpocher, UDAIICEpocher, PrototypeEpocher
from semi_seg.epochers import TrainEpocher, EvalEpocher, UDATrainEpocher, EntropyMinEpocher, MeanTeacherEpocher, \
    IICMeanTeacherEpocher, InferenceEpocher, MIDLPaperEpocher, FeatureOutputCrossIICUDAEpocher, \
    FeatureOutputCrossIICEpocher, InfoNCEEpocher, InfoNCEPretrainEpocher, DifferentiablePrototypeEpocher, \
    UCMeanTeacherEpocher
from semi_seg.epochers.comparable import PICAEpocher

__all__ = ["trainer_zoos"]


class SemiTrainer(Trainer):
    RUN_PATH = str(Path(PROJECT_PATH) / "semi_seg" / "runs")  # noqa
    _only_labeled_data = False
    _train_with_two_stage = False

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
        self.epocher_class = epocher_class  # noqa

    def run_epoch(self, *args, **kwargs):
        self.set_epocher_class()
        epocher = self._run_init()
        return self._run_epoch(epocher, *args, **kwargs)

    def _run_init(self, ):
        epocher = self.epocher_class(
            model=self._model, optimizer=self._optimizer, labeled_loader=self._labeled_loader,
            unlabeled_loader=self._unlabeled_loader, sup_criterion=self._sup_criterion, num_batches=self._num_batches,
            cur_epoch=self._cur_epoch, device=self._device, feature_position=self.feature_positions,
            feature_importance=self._feature_importance
        )
        if self._only_labeled_data:
            epocher.only_with_labeled_data = True
        if self._train_with_two_stage:
            epocher.train_with_two_stage = True
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

    def set_feature_positions(self, feature_positions):
        self.feature_positions = feature_positions  # noqa

    def set_only_labeled_data(self, enable: bool):
        self._only_labeled_data = enable  # noqa

    def set_train_with_two_stage(self, enable: bool):
        self._train_with_two_stage = enable


class FineTuneTrainer(SemiTrainer):

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = TrainEpocher):
        epocher_class.only_with_labeled_data = True
        print(f"Using only labeled data")
        super().set_epocher_class(epocher_class)


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
        self._projector_wrappers = ClusterProjectorWrapper()
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
        self._enforce_matching = config["enforce_matching"]

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = IICTrainEpocher):
        super().set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: IICTrainEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, projectors_wrapper=self._projector_wrappers,
                     IIDSegCriterionWrapper=self._IIDSegWrapper, enforce_matching=self._enforce_matching)
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
                     reg_criterion=self._reg_criterion, enforce_matching=self._enforce_matching)
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


class UCMeanTeacherTrainer(MeanTeacherTrainer):

    def _init(self):
        super()._init()
        self._threshold = RampScheduler(begin_epoch=0, max_epoch=int(self._config["Trainer"]["max_epoch"]) // 3 * 2,
                                        max_value=1, min_value=0.75)

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = UCMeanTeacherEpocher):
        super().set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: UCMeanTeacherEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, teacher_model=self._teacher_model, reg_criterion=self._reg_criterion,
                     ema_updater=self._ema_updater, threshold=self._threshold)
        result = epocher.run()
        self._threshold.step()
        return result


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
                     ema_updater=self._ema_updater, mt_weight=self._mt_weight, iic_weight=self._iic_weight,
                     enforce_matching=self._enforce_matching)
        result = epocher.run()
        return result

    def _eval_epoch(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        evaler = EvalEpocher(self._teacher_model, val_loader=self._val_loader, sup_criterion=self._sup_criterion,
                             cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score


class IICFeatureOutputTrainer(IICTrainer):
    """This class only impose feature output MI"""

    def _init(self):
        super(IICFeatureOutputTrainer, self)._init()
        self._cross_reg_weight = deepcopy(self._reg_weight)
        config = deepcopy(self._config["FeatureOutputIICRegParameters"])
        self._projector_wrappers_output = ClusterProjectorWrapper()
        self._projector_wrappers_output.init_encoder(
            feature_names=self.feature_positions,
        )
        self._projector_wrappers_output.init_decoder(
            feature_names=self.feature_positions,
            **config["DecoderParams"]
        )
        self._IIDSegWrapper_output = IICLossWrapper(
            feature_names=self.feature_positions,
            **config["LossParams"]
        )
        self._output_reg_weight = float(config["weight"])

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = FeatureOutputCrossIICEpocher):
        super().set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: FeatureOutputCrossIICEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(projectors_wrapper=self._projector_wrappers,
                     projectors_wrapper_output=self._projector_wrappers_output,
                     IIDSegCriterionWrapper=self._IIDSegWrapper,
                     IIDSegCriterionWrapper_output=self._IIDSegWrapper_output,
                     cross_reg_weight=self._cross_reg_weight,
                     output_reg_weight=self._output_reg_weight,
                     enforce_matching=self._enforce_matching)

        result = epocher.run()
        return result

    def _init_optimizer(self):
        config = deepcopy(self._config["Optim"])
        self._optimizer = optim.__dict__[config["name"]](
            params=chain(self._model.parameters(), self._projector_wrappers.parameters(),
                         self._projector_wrappers_output.parameters()),
            **{k: v for k, v in config.items() if k != "name"}
        )


class UDAIICFeatureOutputTrainer(UDAIICTrainer):

    def _init(self):
        super()._init()
        config = deepcopy(self._config["FeatureOutputIICRegParameters"])
        self._projector_wrappers_output = ClusterProjectorWrapper()
        self._projector_wrappers_output.init_encoder(
            feature_names=self.feature_positions,
        )
        self._projector_wrappers_output.init_decoder(
            feature_names=self.feature_positions,
            **config["DecoderParams"]
        )
        self._IIDSegWrapper_output = IICLossWrapper(
            feature_names=self.feature_positions,
            **config["LossParams"]
        )
        self._output_reg_weight = float(config["weight"])

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = FeatureOutputCrossIICUDAEpocher):
        super().set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: FeatureOutputCrossIICUDAEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(iic_weight=self._iic_weight, uda_weight=self._uda_weight,
                     output_reg_weight=self._output_reg_weight,
                     projectors_wrapper=self._projector_wrappers,
                     projectors_wrapper_output=self._projector_wrappers_output,
                     reg_criterion=self._reg_criterion, enforce_matching=self._enforce_matching,
                     IIDSegCriterionWrapper=self._IIDSegWrapper,
                     IIDSegCriterionWrapper_output=self._IIDSegWrapper_output)
        result = epocher.run()
        return result


class PICATrainer(SemiTrainer):

    def _init(self):
        super(PICATrainer, self)._init()
        config = deepcopy(self._config["PICARegParameters"])
        self._projector_wrappers = ClusterProjectorWrapper()
        self._projector_wrappers.init_encoder(
            feature_names=self.feature_positions,
            **config["EncoderParams"]
        )
        self._projector_wrappers.init_decoder(
            feature_names=self.feature_positions,
            **config["DecoderParams"]
        )
        self._PICASegWrapper = PICALossWrapper(
            feature_names=self.feature_positions,
            **config["LossParams"]
        )
        self._reg_weight = float(config["weight"])
        self._enforce_matching = config["enforce_matching"]

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = PICAEpocher):
        super().set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: PICAEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, projectors_wrapper=self._projector_wrappers,
                     PICASegCriterionWrapper=self._PICASegWrapper, enforce_matching=self._enforce_matching)
        result = epocher.run()
        return result


class InfoNCETrainer(SemiTrainer):

    def _init(self):
        super()._init()
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
        self._criterion = SupConLoss(**config["LossParams"])
        self._reg_weight = float(config["weight"])

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = InfoNCEEpocher):
        super().set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: InfoNCEEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, projectors_wrapper=self._projector,
                     infoNCE_criterion=self._criterion)
        result = epocher.run()
        return result

    def _init_optimizer(self):
        config = deepcopy(self._config["Optim"])
        self._optimizer = optim.__dict__[config["name"]](
            params=chain(self._model.parameters(), self._projector.parameters()),
            **{k: v for k, v in config.items() if k != "name"}
        )


class InfoNCETrainerDemo(InfoNCETrainer):
    """This training class is going to balance the supervised loss and reg_loss for infonce dynamically to find if
    supervised_regularization framework works."""

    def _init(self):
        # override the InfoNCETrainer
        super(InfoNCETrainer, self)._init()
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
        self._criterion = SupConLoss(**config["LossParams"])

        self._reg_weight = RampScheduler(
            begin_epoch=0, max_epoch=(self._max_epoch // 3) * 2, min_value=float(config["weight"]), max_value=0,
            ramp_mult=-1
        )

    def _run_epoch(self, epocher: InfoNCEEpocher, *args, **kwargs) -> EpochResultDict:
        result = super()._run_epoch(epocher, *args, **kwargs)
        if isinstance(self._reg_weight, WeightScheduler):
            self._reg_weight.step()
        return result


class InfoNCEPretrainTrainer(InfoNCETrainer):
    _only_labeled_data = True

    def _init(self):
        self._config["InfoNCEParameters"]["weight"] = 1.0
        super(InfoNCEPretrainTrainer, self)._init()
        unlabeled_dataset = get_dataset(self._unlabeled_loader)

        dataset = type(unlabeled_dataset)(
            str(Path(unlabeled_dataset._root_dir).parent),
            unlabeled_dataset._mode, unlabeled_dataset._transform
        )

        from torch.utils.data import DataLoader
        from contrastyou.datasets._seg_datset import ContrastBatchSampler  # noqa

        config = self._config

        if config.get("DistributedTrain") is True:
            raise NotImplementedError()

        batch_sampler = ContrastBatchSampler(
            dataset=dataset,
            group_sample_num=config["ContrastData"]["group_sample_num"],
            partition_sample_num=config["ContrastData"]["partition_sample_num"]
        )

        self._contrastive_loader = DataLoader(
            dataset, batch_sampler=batch_sampler,
            num_workers=config["ContrastData"]["num_workers"],
            pin_memory=True
        )

        self._contrastive_loader = iter(self._contrastive_loader)

    def _run_epoch(self, epocher: InfoNCEPretrainEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(chain_dataloader=self._contrastive_loader, projectors_wrapper=self._projector,
                     infoNCE_criterion=self._criterion)
        result = epocher.run()
        return result

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = InfoNCEPretrainEpocher):
        super(InfoNCEPretrainTrainer, self).set_epocher_class(epocher_class)

    def _start_training(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            train_result: EpochResultDict
            eval_result: EpochResultDict
            cur_score: float
            train_result = self.run_epoch()
            # update lr_scheduler
            if hasattr(self, "_scheduler"):
                self._scheduler.step()
            if self.on_master():
                storage_per_epoch = StorageIncomeDict(pretrain=train_result)
                self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
                self._writer.add_scalar_with_StorageDict(storage_per_epoch, self._cur_epoch)
                # save_checkpoint
                self._save_to(self._save_dir, "last.pth")
                # save storage result on csv file.
                self._storage.to_csv(self._save_dir)


class PrototypeTrainer(SemiTrainer):

    def _init(self):
        super()._init()
        config = deepcopy(self._config)["PrototypeParameters"]
        self._projector = ContrastiveProjectorWrapper()
        self._projector.init_encoder(feature_names=self.feature_positions, **config["EncoderParams"])
        self._projector.init_decoder(feature_names=self.feature_positions, **config["DecoderParams"])

        self._register_buffer("memory_bank", dict())
        self._infonce_criterion = SupConLoss(**config["LossParams"])
        self._reg_weight = float(config["weight"])

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = PrototypeEpocher):
        super(PrototypeTrainer, self).set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: PrototypeEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, prototype_projector=self._projector,
                     feature_buffers=self.memory_bank,
                     infoNCE_criterion=self._infonce_criterion)
        result = epocher.run()
        epocher.run_kmeans(self.memory_bank)
        return result


class DifferentiablePrototypeTrainer(SemiTrainer):

    def _init(self):
        super(DifferentiablePrototypeTrainer, self)._init()
        config = deepcopy(self._config)["DPrototypeParameters"]
        self._uda_weight = config["uda_weight"]
        self._cluster_weight = config["cluster_weight"]
        self._prototype_nums = config["prototype_nums"]
        from contrastyou.arch import UNet
        dim = UNet.dimension_dict[self.feature_positions[0]]
        self._prototype_vectors = torch.randn(self._prototype_nums, dim, requires_grad=True,
                                              device=self._device)  # noqa

    def _init_optimizer(self):
        optim_dict = self._config["Optim"]
        self._optimizer = optim.__dict__[optim_dict["name"]](
            params=chain(self._model.parameters(), (self._prototype_vectors,)),
            **{k: v for k, v in optim_dict.items() if k != "name"}
        )

    def set_epocher_class(self, epocher_class: Type[TrainEpocher] = DifferentiablePrototypeEpocher):
        super(DifferentiablePrototypeTrainer, self).set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: DifferentiablePrototypeEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(
            prototype_nums=self._prototype_nums,
            prototype_vectors=self._prototype_vectors,
            cluster_weight=self._cluster_weight,
            uda_weight=self._uda_weight
        )
        result = epocher.run()
        return result


trainer_zoos = {
    "partial": SemiTrainer,
    "uda": UDATrainer,
    "iic": IICTrainer,
    "udaiic": UDAIICTrainer,
    "entropy": EntropyMinTrainer,
    "meanteacher": MeanTeacherTrainer,
    "ucmeanteacher": UCMeanTeacherTrainer,
    "iicmeanteacher": IICMeanTeacherTrainer,
    "midl": MIDLTrainer,
    "featureoutputiic": IICFeatureOutputTrainer,
    "featureoutputudaiic": UDAIICFeatureOutputTrainer,
    "pica": PICATrainer,
    "infonce": InfoNCETrainer,
    "infonce_demo": InfoNCETrainerDemo,
    "infoncepretrain": InfoNCEPretrainTrainer,
    "prototype": PrototypeTrainer,
    "dp": DifferentiablePrototypeTrainer
}
