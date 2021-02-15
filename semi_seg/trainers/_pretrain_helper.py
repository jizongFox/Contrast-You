from functools import partial
from pathlib import Path
from typing import Any, Dict, Callable

from loguru import logger
from torch import nn
from torch.utils.data.dataloader import _BaseDataLoaderIter as BaseDataLoaderIter, DataLoader  # noqa

from contrastyou.arch.unet import freeze_grad
from contrastyou.datasets._seg_datset import ContrastBatchSampler  # noqa
from contrastyou.helper import get_dataset
from deepclustering2.meters2 import EpochResultDict, Storage
from deepclustering2.meters2 import StorageIncomeDict
from deepclustering2.writer import SummaryWriter


def _get_contrastive_dataloader(partial_loader, config):
    # going to get all dataset with contrastive sampler

    unlabeled_dataset = get_dataset(partial_loader)

    dataset = type(unlabeled_dataset)(
        str(Path(unlabeled_dataset._root_dir).parent),  # noqa
        unlabeled_dataset._mode, unlabeled_dataset._transform  # noqa
    )

    contrastive_config = config["ContrastiveLoaderParams"]
    num_workers = contrastive_config.pop("num_workers")
    batch_sampler = ContrastBatchSampler(
        dataset=dataset,
        **contrastive_config
    )
    contrastive_loader = DataLoader(
        dataset, batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    return iter(contrastive_loader)


class _PretrainTrainerMixin:
    _model: nn.Module
    _unlabeled_loader: iter
    _config: Dict[str, Any]
    _start_epoch: int
    _max_epoch: int
    _save_dir: str
    init: Callable[..., None]
    on_master: Callable[[], bool]
    run_epoch: Callable[[], EpochResultDict]
    _save_to: Callable[[str, str], None]
    _contrastive_loader = BaseDataLoaderIter
    _storage: Storage
    _writer: SummaryWriter

    def __init__(self, *args, **kwargs):
        super(_PretrainTrainerMixin, self).__init__(*args, **kwargs)
        self.__initialized_grad = False

    def _init(self, *args, **kwargs):
        super(_PretrainTrainerMixin, self)._init(*args, **kwargs)  # noqa
        # here you have conventional training objects
        self._contrastive_loader = _get_contrastive_dataloader(self._unlabeled_loader, self._config)
        logger.debug("creating contrastive_loader")

    def _run_epoch(self, epocher, *args, **kwargs) -> EpochResultDict:
        epocher.init = partial(epocher.init, chain_dataloader=self._contrastive_loader, )
        return super(_PretrainTrainerMixin, self)._run_epoch(epocher, *args, **kwargs)  # noqa

    def enable_grad(self, from_, util_):
        self.__from = from_
        self.__util = util_
        self.__initialized_grad = True
        logger.info("set grad from {} to {}", from_, util_)
        return freeze_grad(self._model, from_=self.__from, util_=self.__util)  # noqa

    def _start_training(self):
        assert self.__initialized_grad, "`enable_grad` must be called first"
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
