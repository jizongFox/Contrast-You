from functools import partial
from typing import Dict, Any, Callable, Type

from torch import nn
from torch.utils.data.dataloader import _BaseDataLoaderIter as BaseDataLoaderIter, DataLoader  # noqa

from contrastyou.epochers.base import EpocherBase
from contrastyou.meters import Storage
from contrastyou.writer import SummaryWriter
from semi_seg.epochers.new_pretrain import PretrainEpocher
from semi_seg.trainers._helper import _get_contrastive_dataloader
from semi_seg.trainers.new_trainer import SemiTrainer


class _PretrainTrainerMixin:
    _model: nn.Module
    _unlabeled_loader: iter
    _config: Dict[str, Any]
    _start_epoch: int
    _max_epoch: int
    _save_dir: str
    init: Callable[..., None]
    on_master: Callable[[], bool]
    run_tra_epoch: Callable
    save_to: Callable
    _contrastive_loader: BaseDataLoaderIter
    _storage: Storage
    _writer: SummaryWriter
    activate_hooks = True

    def __init__(self, inference_until: str, **kwargs):
        super(_PretrainTrainerMixin, self).__init__(**kwargs)
        self._contrastive_loader, self._monitor_loader = _get_contrastive_dataloader(
            self._unlabeled_loader, self._config["ContrastiveLoaderParams"]
        )
        self._inference_until = inference_until

    def _run_epoch(self, epocher, *args, **kwargs):
        epocher.init = partial(epocher.init, chain_dataloader=self._contrastive_loader,
                               monitor_dataloader=self._monitor_loader)
        return super(_PretrainTrainerMixin, self)._run_epoch(epocher, *args, **kwargs)  # noqa

    def _start_training(self, **kwargs):
        start_epoch = max(self._cur_epoch + 1, self._start_epoch)
        self._cur_score: float

        for self._cur_epoch in range(start_epoch, self._max_epoch):
            with self._storage:  # save csv each epoch
                train_metrics = self.run_tra_epoch()
                if self.on_master():
                    self._storage.add_from_meter_interface(
                        pre_tra=train_metrics, epoch=self._cur_epoch)
                    self._writer.add_scalars_from_meter_interface(
                        pre_tra=train_metrics, epoch=self._cur_epoch)

                if hasattr(self, "_scheduler"):
                    self._scheduler.step()

                self.save_to(save_name="last.pth")

    @property
    def train_epocher(self) -> Type[EpocherBase]:
        return PretrainEpocher

    def _create_tra_epoch(self, **kwargs) -> EpocherBase:
        epocher = self.train_epocher(
            model=self._model, optimizer=self._optimizer, labeled_loader=self._labeled_loader,
            unlabeled_loader=self._unlabeled_loader, sup_criterion=self._criterion, num_batches=self._num_batches,
            cur_epoch=self._cur_epoch, device=self._device, two_stage=self._two_stage, disable_bn=self._disable_bn,
            chain_dataloader=self._contrastive_loader, inference_until=self._inference_until,
        )
        if self.activate_hooks:
            if len(self.__hooks__) > 0:
                epocher.add_hooks([h() for h in self.__hooks__])
        epocher.init()
        return epocher


class PretrainTrainer(_PretrainTrainerMixin, SemiTrainer):
    pass