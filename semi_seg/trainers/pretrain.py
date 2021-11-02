from functools import partial
from typing import Dict, Any, Callable, Type, Union, Optional

from loguru import logger
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from torch.nn import ModuleList
from torch.optim import Optimizer
from torch.utils.data.dataloader import _BaseDataLoaderIter as BaseDataLoaderIter, DataLoader  # noqa

from contrastyou.arch import UNet
from contrastyou.losses import LossClass
from contrastyou.meters import Storage
from contrastyou.types import SizedIterable
from contrastyou.writer import SummaryWriter
from semi_seg.epochers.epocher import EpocherBase
from semi_seg.epochers.pretrain import PretrainEncoderEpocher, PretrainDecoderEpocher
from semi_seg.trainers._helper import _get_contrastive_dataloader
from semi_seg.trainers.trainer import SemiTrainer


class _PretrainTrainerMixin:
    _model: nn.Module
    _labeled_loader: SizedIterable
    _unlabeled_loader: SizedIterable
    _config: Dict[str, Any]
    _start_epoch: int
    _max_epoch: int
    _save_dir: str
    init: Callable[..., None]
    on_master: Callable[[], bool]
    tra_epoch: Callable
    save_to: Callable
    _contrastive_loader: BaseDataLoaderIter
    _storage: Storage
    _writer: Optional[SummaryWriter]
    activate_hooks = True
    __hooks__: ModuleList
    _optimizer: Optimizer
    train_epocher: Type[EpocherBase]
    _cur_epoch: int
    _device: str
    _num_batches: int
    scaler: GradScaler
    _criterion: LossClass[Tensor]

    def __init__(self, **kwargs):
        super(_PretrainTrainerMixin, self).__init__(**kwargs)
        if "ContrastiveLoaderParams" not in self._config:
            raise RuntimeError(
                f"`ContrastiveLoaderParams` should be found in config, given \n`{', '.join(self._config.keys())}`"
            )

        self._contrastive_loader, self._monitor_loader = _get_contrastive_dataloader(
            self._unlabeled_loader, self._config["ContrastiveLoaderParams"]
        )
        self._inference_until = None

    @property
    def forward_until(self) -> str:
        if self._inference_until is None:
            return list(UNet.decoder_names)[-1]
        return self._inference_until

    @forward_until.setter
    def forward_until(self, forward_until: Union[str, None]):
        if isinstance(forward_until, str):
            if forward_until == "all":
                self._inference_until = None
                logger.opt(depth=1).debug(f"{self.__class__.__name__} set forward pass to {self.forward_until}")
                return
            assert forward_until in UNet.arch_elements, forward_until
        self._inference_until = forward_until
        logger.opt(depth=1).debug(f"{self.__class__.__name__} set forward pass to {self.forward_until}")

    def _run_epoch(self, epocher, *args, **kwargs):
        epocher.init = partial(epocher.init, chain_dataloader=self._contrastive_loader,
                               monitor_dataloader=self._monitor_loader)
        return super(_PretrainTrainerMixin, self)._run_epoch(epocher, *args, **kwargs)  # noqa

    def _start_training(self, **kwargs):
        start_epoch = max(self._cur_epoch + 1, self._start_epoch)
        self._cur_score: float

        for self._cur_epoch in range(start_epoch, self._max_epoch + 1):
            with self._storage:  # save csv each epoch
                train_metrics = self.tra_epoch()
                if self.on_master():
                    self._storage.add_from_meter_interface(
                        pre_tra=train_metrics, epoch=self._cur_epoch)
                    self._writer.add_scalars_from_meter_interface(
                        pre_tra=train_metrics, epoch=self._cur_epoch)

                if hasattr(self, "_scheduler"):
                    self._scheduler.step()

            if self.on_master():
                self.save_to(save_name="last.pth")

    def _create_initialized_tra_epoch(self, **kwargs) -> EpocherBase:
        epocher = self.train_epocher(
            model=self._model, optimizer=self._optimizer, labeled_loader=self._labeled_loader,
            unlabeled_loader=self._unlabeled_loader, sup_criterion=self._criterion, num_batches=self._num_batches,
            cur_epoch=self._cur_epoch, device=self._device, two_stage=False, disable_bn=False,
            chain_dataloader=self._contrastive_loader, inference_until=self._inference_until, scaler=self.scaler,
            accumulate_iter=1
        )
        epocher.set_trainer(self)
        epocher.init()
        return epocher


class PretrainEncoderTrainer(_PretrainTrainerMixin, SemiTrainer):
    @property
    def train_epocher(self) -> Type[EpocherBase]:
        return PretrainEncoderEpocher


class PretrainDecoderTrainer(_PretrainTrainerMixin, SemiTrainer):
    @property
    def train_epocher(self) -> Type[EpocherBase]:
        return PretrainDecoderEpocher
