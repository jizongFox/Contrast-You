import os
from abc import abstractmethod
from contextlib import contextmanager, nullcontext
from itertools import chain
from pathlib import Path
from typing import Dict, Any, Optional, cast

import torch.optim.optimizer
from loguru import logger
from torch import Tensor, nn, optim

from contrastyou.meters import Storage
from ._amp import AMPScalerMixin
from ._ddp import DDPMixin
from ._hooks import HookMixin
from ._io import IOMixin
from .. import MODEL_PATH, success
from ..epochers.base import EpocherBase
from ..losses import LossClass
from ..nn import ModuleBase, Buffer
from ..optim import GradualWarmupScheduler
from ..trainer.base import run_once
from ..types import SizedIterable
from ..writer import SummaryWriter


class Trainer(IOMixin, HookMixin, DDPMixin, AMPScalerMixin, ModuleBase):
    RUN_PATH = MODEL_PATH  # type:str # absolute path

    def __init__(self, *, model: nn.Module, criterion: LossClass[Tensor], tra_loader: SizedIterable,
                 val_loader: SizedIterable, save_dir: str, max_epoch: int = 100, num_batches: int = 100, device="cpu",
                 config: Dict[str, Any], enable_scale: bool = False) -> None:
        super().__init__(enable_scale=enable_scale, accumulate_iter=1)
        self._initialized = False

        self._model = self._inference_model = model
        self._criterion = criterion
        self._tra_loader = tra_loader
        self._val_loader = val_loader
        self._save_dir = cast(str, Buffer(save_dir))
        self._max_epoch = cast(int, Buffer(max_epoch))
        self._num_batches = Buffer(num_batches)
        self._device = device
        self.config = cast(dict, Buffer(config))
        self._config = self.config

        self._storage = Storage(save_dir=self._save_dir)
        self._writer = SummaryWriter(log_dir=self._save_dir) if self.on_master else None

        self._optimizer = None
        self._scheduler = None
        self._cur_epoch = Buffer(0)
        self._start_epoch = Buffer(0)
        self._best_score = Buffer(0.0)

    def init(self):
        if self._initialized:
            raise RuntimeError(f"{self.__class__.__name__} has been initialized.")
        self._optimizer = self._init_optimizer()
        self._scheduler = self._init_scheduler(self._optimizer, scheduler_params=self._config.get("Scheduler", None))
        self._initialized = True

    def _init_optimizer(self) -> torch.optim.Optimizer:
        optim_params = self._config["Optim"]
        optimizer = optim.__dict__[optim_params["name"]](
            params=filter(lambda p: p.requires_grad, self._model.parameters()),
            **{k: v for k, v in optim_params.items() if k not in ["name", "pre_lr", "ft_lr"]})

        optimizer.add_param_group({"params": chain(*(x.parameters() for x in self._hooks)),
                                   **{k: v for k, v in optim_params.items() if k not in ["name", "pre_lr", "ft_lr"]}})

        return optimizer

    def _init_scheduler(self, optimizer, scheduler_params) -> Optional[GradualWarmupScheduler]:
        if scheduler_params is None:
            return None
        max_epoch = self._max_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epoch - int(self._config["Scheduler"]["warmup_max"]),
            eta_min=1e-7
        )
        scheduler = GradualWarmupScheduler(optimizer, scheduler_params["multiplier"],
                                           total_epoch=scheduler_params["warmup_max"],
                                           after_scheduler=scheduler)
        return scheduler

    def start_training(self, **kwargs):
        if not self._initialized:
            raise RuntimeError(f"{self.__class__.__name__} should call `init()` first")
        self.to(self.device)
        with self._writer if self.on_master else nullcontext():
            self._start_training(**kwargs)
            if self.on_master:
                success(save_dir=self.absolute_save_dir)

    def _start_training(self, **kwargs):
        start_epoch = max(self._cur_epoch + 1, self._start_epoch)
        self._cur_score: float

        for self._cur_epoch in range(start_epoch, self._max_epoch + 1):
            with self._storage:  # save csv each epoch
                train_metrics = self.tra_epoch()
                if self.on_master:
                    eval_metrics, cur_score = self.eval_epoch(model=self.inference_model, loader=self._val_loader)
                    test_metrics, _ = self.eval_epoch(model=self.inference_model, loader=self._test_loader)

                    self._storage.add_from_meter_interface(tra=train_metrics, val=eval_metrics, test=test_metrics,
                                                           epoch=self._cur_epoch)
                    self._writer.add_scalars_from_meter_interface(tra=train_metrics, val=eval_metrics,
                                                                  test=test_metrics, epoch=self._cur_epoch)

                if hasattr(self, "_scheduler"):
                    self._scheduler.step()

                best_case_sofa = self._best_score < cur_score
                if best_case_sofa:
                    self._best_score = cur_score

            if self.on_master:
                self.save_to(save_name="last.pth")
                if best_case_sofa:
                    self.save_to(save_name="best.pth")

    def tra_epoch(self, **kwargs):
        epocher = self._create_initialized_tra_epoch(**kwargs)
        return self._run_tra_epoch(epocher)

    def _run_tra_epoch(self, epocher: EpocherBase):
        use_hook = self.activate_hooks and len(self._hooks) > 0
        with epocher.register_hook(*[h() for h in self._hooks]) if use_hook else nullcontext():
            epocher.run()
        return epocher.get_metric()

    @abstractmethod
    def _create_initialized_tra_epoch(self, **kwargs) -> EpocherBase:
        ...

    def eval_epoch(self, *, model, loader, **kwargs):
        epocher = self._create_initialized_eval_epoch(model=model, loader=loader, **kwargs)
        return self._run_eval_epoch(epocher)

    @abstractmethod
    def _create_initialized_eval_epoch(self, *, model, loader, **kwargs) -> EpocherBase:
        ...

    def _run_eval_epoch(self, epocher):
        epocher.run()
        return epocher.get_metric(), epocher.get_score()

    @property
    def inference_model(self):
        return self._inference_model

    @run_once
    def set_model4inference(self, model: nn.Module):
        logger.trace(f"change inference model from {id(self._inference_model)} to {id(model)}")
        self._inference_model = model

    @contextmanager
    def switch_inference_model(self, model: nn.Module):
        previous_ = self.inference_model
        self.set_model4inference(model)
        yield
        self.set_model4inference(previous_)

    @property
    def save_dir(self) -> str:
        """return absolute save_dir """
        return str(self._save_dir)

    @property
    def absolute_save_dir(self) -> str:
        return self.save_dir

    @property
    def relative_save_dir(self):
        """return relative save_dir, raise error if the save_dir is given as absolute path in the __init__
        and not relative to model_path"""
        return str(Path(self.absolute_save_dir).relative_to(self.RUN_PATH))

    @property
    def success(self):
        return ".success" in os.listdir(self.absolute_save_dir)

    @property
    def device(self):
        return self._device
