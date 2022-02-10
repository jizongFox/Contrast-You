import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Type, Dict, Any

from loguru import logger
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from contrastyou import optim
from contrastyou.arch.discriminator import Discriminator, class_name
from contrastyou.data import ScanBatchSampler
from contrastyou.losses import LossClass
from contrastyou.trainer import create_save_dir
from contrastyou.trainer.base import Trainer
from contrastyou.types import criterionType, SizedIterable
from contrastyou.utils import fix_all_seed_within_context, get_dataset
from contrastyou.utils.printable import item2str
from semi_seg.epochers.comparable import MixUpEpocher, AdversarialEpocher
from semi_seg.epochers.epocher import EpocherBase, SemiSupervisedEpocher, FineTuneEpocher, EvalEpocher, DMTEpcoher, \
    InferenceEpocher
from semi_seg.hooks import MeanTeacherTrainerHook, EMAUpdater


class SemiTrainer(Trainer):
    activate_hooks = True

    def __init__(self, *, model: nn.Module, labeled_loader: SizedIterable, unlabeled_loader: SizedIterable,
                 val_loader: SizedIterable, test_loader: SizedIterable, criterion: LossClass[Tensor], save_dir: str,
                 max_epoch: int = 100, num_batches: int = 100, device="cpu", disable_bn: bool, two_stage: bool,
                 config: Dict[str, Any], enable_scale=True, accumulate_iter: int = 1, **kwargs) -> None:
        super().__init__(model=model, criterion=criterion, tra_loader=None, val_loader=val_loader,  # noqa
                         save_dir=save_dir, max_epoch=max_epoch, num_batches=num_batches, device=device, config=config,
                         **kwargs)
        del self._tra_loader
        self._labeled_loader: SizedIterable = labeled_loader
        self._unlabeled_loader: SizedIterable = unlabeled_loader
        self._val_loader: SizedIterable = val_loader
        self._test_loader: SizedIterable = test_loader
        self._sup_criterion = criterion
        self._disable_bn = disable_bn
        self._two_stage = two_stage
        self._enable_scale = enable_scale
        self.scaler = GradScaler(enabled=enable_scale)
        self._accumulate_iter = accumulate_iter
        logger.info(f"{'Enable' if enable_scale else 'Disable'} mixed precision training "
                    f"with an accumulate iter: {accumulate_iter}")

    @property
    def train_epocher(self) -> Type[EpocherBase]:
        return SemiSupervisedEpocher

    def _create_initialized_tra_epoch(self, **kwargs) -> EpocherBase:
        epocher = self.train_epocher(
            model=self._model, optimizer=self._optimizer, labeled_loader=self._labeled_loader,
            unlabeled_loader=self._unlabeled_loader, sup_criterion=self._criterion, num_batches=self._num_batches,
            cur_epoch=self._cur_epoch, device=self._device, two_stage=self._two_stage, disable_bn=self._disable_bn,
            scaler=self.scaler, accumulate_iter=self._accumulate_iter
        )
        epocher.set_trainer(self)
        epocher.init()
        return epocher

    def _create_initialized_eval_epoch(self, *, model, loader, **kwargs) -> EpocherBase:
        epocher = EvalEpocher(model=model, loader=loader, sup_criterion=self._criterion, cur_epoch=self._cur_epoch,
                              device=self._device, scaler=self.scaler, accumulate_iter=self._accumulate_iter)
        epocher.set_trainer(self)
        epocher.init()
        return epocher

    def inference(self, checkpoint_path: str = None, checkpoint_name: str = "best.pth", save_dir: str = None):
        # make self._test_loader to be a patient based Dataloader
        # load_checkpoint
        if checkpoint_path is None:
            checkpoint_path = self.absolute_save_dir
        if not os.path.isabs(checkpoint_path):
            raise ValueError(f"`checkpoint_path` must be an absolute path, given {checkpoint_path}")

        logger.info(f"Resume checkpoint from {checkpoint_path}...")
        self.resume_from_path(str(checkpoint_path), name=checkpoint_name)

        test_loader = self.patch_scan_based_dataloader(self._test_loader)

        epoch_metric, _ = self._inference(test_loader=test_loader)

        if save_dir is None:
            save_dir = self.absolute_save_dir
        else:
            if not os.path.isabs(save_dir):
                save_dir = create_save_dir(self, save_dir)

        assert save_dir
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(save_dir, "inference_result.json"), "w") as f:
            json.dump(epoch_metric, f, indent=4)
            logger.info(f"Saved inference results in {save_dir}/\"inference_result.json\"")

        if self._writer:
            self._writer.add_scalars_from_meter_interface(infer=epoch_metric, epoch=10000)
        logger.info(f"Inference results: " + item2str(epoch_metric))
        logger.success(f"{class_name(self)} Done")

    # store the results.

    def _inference(self, *, test_loader: DataLoader):
        epocher = InferenceEpocher(model=self._model, loader=test_loader, sup_criterion=self._criterion,
                                   cur_epoch=10000, device=self._device, scaler=self.scaler,
                                   accumulate_iter=self._accumulate_iter)
        epocher.set_trainer(self)
        epocher.init()
        epocher.run()
        return epocher.get_metric(), epocher.get_score()

    @staticmethod
    def patch_scan_based_dataloader(loader) -> DataLoader:

        # redo the test_loader
        test_dataset = get_dataset(loader)
        patient_sampler = ScanBatchSampler(dataset=test_dataset, shuffle=False)
        loader = DataLoader(test_dataset, batch_sampler=patient_sampler, batch_size=1, num_workers=2)
        return loader


class MTTrainer(SemiTrainer):
    def _start_training(self, **kwargs):
        start_epoch = max(self._cur_epoch + 1, self._start_epoch)
        self._cur_score: float

        mt_hook = [h for h in self._hooks if isinstance(h, MeanTeacherTrainerHook)]
        assert len(mt_hook) == 1, mt_hook
        mt_hook = mt_hook[0]

        for self._cur_epoch in range(start_epoch, self._max_epoch + 1):
            with self._storage:  # save csv each epoch
                train_metrics = self.tra_epoch()
                if self.on_master():
                    logger.info("inference on teacher model")
                    with self.switch_inference_model(mt_hook.teacher_model):
                        eval_metrics, cur_score = self.eval_epoch(model=self.inference_model, loader=self._val_loader)
                        test_metrics, _________ = self.eval_epoch(model=self.inference_model, loader=self._test_loader)
                    extra_result = {}
                    for i, teacher in enumerate(mt_hook.extra_teachers):
                        logger.info(f"inference on extra teacher model {i}")
                        with self.switch_inference_model(teacher):
                            eval_metrics, _ = self.eval_epoch(model=self.inference_model, loader=self._val_loader)
                            test_metrics, _ = self.eval_epoch(model=self.inference_model, loader=self._test_loader)
                        extra_result[f"eval_extra_teacher_{i}"] = eval_metrics
                        extra_result[f"test_extra_teacher_{i}"] = test_metrics

                    self._storage.add_from_meter_interface(tra=train_metrics, val=eval_metrics, test=test_metrics,
                                                           epoch=self._cur_epoch, **extra_result)
                    self._writer.add_scalars_from_meter_interface(tra=train_metrics, val=eval_metrics,
                                                                  test=test_metrics, epoch=self._cur_epoch,
                                                                  **extra_result)

                if hasattr(self, "_scheduler"):
                    self._scheduler.step()

                best_case_sofa = self._best_score < cur_score
                if best_case_sofa:
                    self._best_score = cur_score

            if self.on_master():
                self.save_to(save_name="last.pth")
                if best_case_sofa:
                    self.save_to(save_name="best.pth")


class DMTTrainer(SemiTrainer):

    def __init__(self, *, model: nn.Module, labeled_loader: SizedIterable, unlabeled_loader: SizedIterable,
                 val_loader: SizedIterable, test_loader: SizedIterable, criterion: LossClass[Tensor], save_dir: str,
                 max_epoch: int = 100, num_batches: int = 100, device="cpu", disable_bn: bool, two_stage: bool,
                 config: Dict[str, Any], enable_scale=True, accumulate_iter: int = 1, **kwargs) -> None:
        super().__init__(model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
                         val_loader=val_loader, test_loader=test_loader, criterion=criterion, save_dir=save_dir,
                         max_epoch=max_epoch, num_batches=num_batches, device=device, disable_bn=disable_bn,
                         two_stage=two_stage, config=config, enable_scale=enable_scale, accumulate_iter=accumulate_iter,
                         **kwargs)
        self._teacher_model = deepcopy(model)

    @property
    def train_epocher(self) -> Type[DMTEpcoher]:
        return DMTEpcoher

    def _create_initialized_tra_epoch(self, **kwargs) -> EpocherBase:
        epocher = self.train_epocher(
            model=self._model, optimizer=self._optimizer, labeled_loader=self._labeled_loader,
            unlabeled_loader=self._unlabeled_loader, sup_criterion=self._criterion, num_batches=self._num_batches,
            cur_epoch=self._cur_epoch, device=self._device, two_stage=self._two_stage, disable_bn=self._disable_bn,
            scaler=self.scaler, accumulate_iter=self._accumulate_iter, mt_criterion=nn.MSELoss(),
            ema_updater=EMAUpdater(), teacher_model=self._teacher_model
        )
        epocher.init()
        return epocher


class FineTuneTrainer(SemiTrainer):
    activate_hooks = False

    @property
    def train_epocher(self) -> Type[EpocherBase]:
        return FineTuneEpocher


class MixUpTrainer(SemiTrainer):
    activate_hooks = True

    @property
    def train_epocher(self) -> Type[EpocherBase]:
        return MixUpEpocher


class AdversarialTrainer(SemiTrainer):
    """
    adversarial trainer for medical images, without using hooks.
    """
    activate_hooks = False

    def __init__(self, *, model: nn.Module, labeled_loader: SizedIterable, unlabeled_loader: SizedIterable,
                 val_loader: SizedIterable, test_loader: SizedIterable, criterion: criterionType, save_dir: str,
                 max_epoch: int = 100, num_batches: int = 100, device="cpu", disable_bn: bool, two_stage: bool,
                 config: Dict[str, Any],
                 reg_weight: int, dis_consider_image: bool = False, **kwargs) -> None:
        super().__init__(model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
                         val_loader=val_loader, test_loader=test_loader, criterion=criterion, save_dir=save_dir,
                         max_epoch=max_epoch, num_batches=num_batches, device=device, disable_bn=disable_bn,
                         two_stage=two_stage, config=config, **kwargs)
        input_dim = self._model._input_dim + self._model.num_classes if dis_consider_image else self._model.num_classes
        self._dis_consider_image = dis_consider_image
        logger.trace(f"Initializing the discriminator with input_dim = {input_dim}")
        seed = self._config.get("RandomSeed", 10)
        with fix_all_seed_within_context(seed):
            self._discriminator = Discriminator(input_dim=input_dim, hidden_dim=64)
        optim_params = self._config["Optim"]
        logger.trace(
            f'Initializing the discriminator optimizer with '
            f'{item2str({k: v for k, v in optim_params.items() if k != "name" and k != "pre_lr" and k != "ft_lr"})}'
        )
        self._dis_optimizer = optim.__dict__[optim_params["name"]](
            params=filter(lambda p: p.requires_grad, self._discriminator.parameters()),
            **{k: v for k, v in optim_params.items() if k != "name" and k != "pre_lr" and k != "ft_lr"}
        )
        self._reg_weight = float(reg_weight)
        logger.trace(f"Initializing weight = {float(self._reg_weight)}")

    @property
    def train_epocher(self) -> Type[EpocherBase]:
        return AdversarialEpocher

    def _create_initialized_tra_epoch(self, **kwargs) -> EpocherBase:
        epocher = self.train_epocher(
            model=self._model, optimizer=self._optimizer, labeled_loader=self._labeled_loader,
            unlabeled_loader=self._unlabeled_loader, sup_criterion=self._criterion, num_batches=self._num_batches,
            cur_epoch=self._cur_epoch, device=self._device, two_stage=self._two_stage, disable_bn=self._disable_bn,
            discriminator=self._discriminator, discr_optimizer=self._dis_optimizer, reg_weight=self._reg_weight,
            dis_consider_image=self._dis_consider_image, scaler=self.scaler
        )
        epocher.init()
        return epocher
