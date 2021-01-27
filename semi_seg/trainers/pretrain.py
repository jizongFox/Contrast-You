from functools import partial
from pathlib import Path
from typing import Any, Dict

from contrastyou.helper import get_dataset
from deepclustering2.meters2 import EpochResultDict
from deepclustering2.meters2 import StorageIncomeDict
from .trainer import InfoNCETrainer, IICTrainer, UDAIICTrainer


def _get_contrastive_dataloader(partial_loader, config):
    unlabeled_dataset = get_dataset(partial_loader)

    dataset = type(unlabeled_dataset)(
        str(Path(unlabeled_dataset._root_dir).parent),
        unlabeled_dataset._mode, unlabeled_dataset._transform
    )

    from torch.utils.data import DataLoader
    from contrastyou.datasets._seg_datset import ContrastBatchSampler  # noqa

    if config.get("DistributedTrain") is True:
        raise NotImplementedError()
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
    _unlabeled_loader: iter
    _config: Dict[str, Any]

    def init(self, *args, **kwargs):
        super(_PretrainTrainerMixin, self).init(*args, **kwargs)
        # here you have conventional training objects
        self._contrastive_loader = _get_contrastive_dataloader(self._unlabeled_loader, self._config)

    def _run_epoch(self, epocher, *args, **kwargs) -> EpochResultDict:
        epocher.init = partial(epocher.init, chain_dataloader=self._contrastive_loader, )
        return super(_PretrainTrainerMixin, self)._run_epoch(epocher, *args, **kwargs)

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


class PretrainInfoNCETrainer(_PretrainTrainerMixin, InfoNCETrainer):
    from semi_seg.epochers.pretrain import InfoNCEPretrainEpocher

    def _set_epocher_class(self, epocher_class=InfoNCEPretrainEpocher):
        super(PretrainInfoNCETrainer, self)._set_epocher_class(epocher_class)


class PretrainIICTrainer(_PretrainTrainerMixin, IICTrainer):
    from semi_seg.epochers.pretrain import MIPretrainEpocher

    def _set_epocher_class(self, epocher_class=MIPretrainEpocher):
        super(PretrainIICTrainer, self)._set_epocher_class(epocher_class)


class PretrainUDAIICTrainer(_PretrainTrainerMixin, UDAIICTrainer):
    from semi_seg.epochers.pretrain import UDAIICPretrainEpocher

    def _set_epocher_class(self, epocher_class=UDAIICPretrainEpocher):
        super(PretrainUDAIICTrainer, self)._set_epocher_class(epocher_class)


from .newtrainer import ExperimentalTrainer
from ..epochers.pretrain import ExperimentalPretrainEpocher


class PretrainExperimentalTrainer(_PretrainTrainerMixin, ExperimentalTrainer):
    def _set_epocher_class(self, epocher_class=ExperimentalPretrainEpocher):
        super(PretrainExperimentalTrainer, self)._set_epocher_class(epocher_class)
