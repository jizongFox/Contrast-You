from itertools import chain
from typing import Dict, Any, Optional, cast

import torch.optim.optimizer
from torch import Tensor, nn, optim

from contrastyou.meters import Storage
from ._amp import AMPScalerMixin
from ._base import _TrainerBase, Buffer
from ._ddp import DDPMixin
from ._hooks import _HookMixin
from .. import MODEL_PATH
from ..losses import LossClass
from ..optim import GradualWarmupScheduler
from ..types import SizedIterable
from ..writer import SummaryWriter


class Trainer(_HookMixin, DDPMixin, AMPScalerMixin, _TrainerBase):
    RUN_PATH = MODEL_PATH  # type:str # absolute path

    def __init__(self, *, model: nn.Module, criterion: LossClass[Tensor], tra_loader: SizedIterable,
                 val_loader: SizedIterable, save_dir: str, max_epoch: int = 100, num_batches: int = 100, device="cpu",
                 config: Dict[str, Any], enable_scale: bool = False) -> None:
        super().__init__(enable_scale=enable_scale, accumulate_iter=1)
        self._initialized = False

        self._model = model
        self.criterion = criterion
        self._tra_loader = tra_loader
        self._val_loader = val_loader
        self.save_dir = cast(str, Buffer(save_dir))
        self._save_dir = self.save_dir
        self._max_epoch = Buffer(max_epoch)
        self._num_batches = Buffer(num_batches)
        self.device = device
        self.config = Buffer(config)
        self._config = self.config

        self._storage = Storage(save_dir=self._save_dir)
        self._writer = SummaryWriter(log_dir=self._save_dir) if self.on_master else None

        self._optimizer = None
        self._scheduler = None

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
            T_max=max_epoch - self._config["Scheduler"]["warmup_max"],
            eta_min=1e-7
        )
        scheduler = GradualWarmupScheduler(optimizer, scheduler_params["multiplier"],
                                           total_epoch=scheduler_params["warmup_max"],
                                           after_scheduler=scheduler)
        return scheduler