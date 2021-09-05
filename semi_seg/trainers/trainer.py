from typing import Type, Dict, Any

from loguru import logger
from torch import nn
from torch.cuda.amp import GradScaler

from contrastyou import optim
from contrastyou.arch.discriminator import Discriminator
from contrastyou.losses.kl import KL_div
from contrastyou.trainer.base import Trainer
from contrastyou.types import criterionType
from contrastyou.utils import fix_all_seed_within_context
from contrastyou.utils.printable import item2str
from semi_seg.epochers.comparable import MixUpEpocher, AdversarialEpocher
from semi_seg.epochers.epocher import EpocherBase, SemiSupervisedEpocher, FineTuneEpocher, EvalEpocher
from semi_seg.helper import SizedIterable


class SemiTrainer(Trainer):
    activate_hooks = True

    def __init__(self, *, model: nn.Module, labeled_loader: SizedIterable, unlabeled_loader: SizedIterable,
                 val_loader: SizedIterable, test_loader: SizedIterable, criterion: KL_div, save_dir: str,
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
        epocher.init()
        return epocher

    def _create_initialized_eval_epoch(self, *, model, loader, **kwargs) -> EpocherBase:
        epocher = EvalEpocher(model=model, loader=loader, sup_criterion=self._criterion, cur_epoch=self._cur_epoch,
                              device=self._device, scaler=self.scaler, accumulate_iter=self._accumulate_iter)
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
