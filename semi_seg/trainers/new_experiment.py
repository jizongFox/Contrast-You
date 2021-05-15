from typing import Type, Dict, Any

from deepclustering2.type import T_loader
from torch import nn

from contrastyou.trainer.base import Trainer
from contrastyou.types import criterionType as _criterion_type
from semi_seg.epochers.new_epocher import EpocherBase, SemiSupervisedEpocher, FineTuneEpocher, EvalEpocher


class SemiTrainer(Trainer):

    def __init__(self, *, model: nn.Module, labeled_loader: T_loader, unlabeled_loader: T_loader, val_loader: T_loader,
                 test_loader: T_loader, criterion: _criterion_type, save_dir: str, max_epoch: int = 100,
                 num_batches: int = 100, device="cpu", disable_bn: bool, two_stage: bool,
                 config: Dict[str, Any], **kwargs) -> None:
        super().__init__(model=model, criterion=criterion, tra_loader=None, val_loader=val_loader,
                         save_dir=save_dir, max_epoch=max_epoch, num_batches=num_batches, device=device, config=config,
                         **kwargs)
        del self._tra_loader
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._val_loader = val_loader
        self._test_loader = test_loader
        self._sup_criterion = criterion
        self._disable_bn = disable_bn
        self._two_stage = two_stage

    @property
    def train_epocher(self) -> Type[EpocherBase]:
        return SemiSupervisedEpocher

    def _create_tra_epoch(self, **kwargs) -> EpocherBase:
        epocher = self.train_epocher(
            model=self._model, optimizer=self._optimizer, labeled_loader=self._labeled_loader,
            unlabeled_loader=self._unlabeled_loader, sup_criterion=self._criterion, num_batches=self._num_batches,
            cur_epoch=self._cur_epoch, device=self._device, two_stage=self._two_stage, disable_bn=self._disable_bn
        )
        if len(self.__hooks__) > 0:
            epocher.add_hooks([h() for h in self.__hooks__])
        epocher.init()
        return epocher

    def _create_eval_epoch(self, *, model, loader, **kwargs) -> EpocherBase:
        epocher = EvalEpocher(model=model, loader=loader, sup_criterion=self._criterion, cur_epoch=self._cur_epoch,
                              device=self._device)
        epocher.init()
        return epocher


class FineTuneTrainer(SemiTrainer):

    def __init__(self, *, model: nn.Module, labeled_loader: T_loader, val_loader: T_loader,
                 test_loader: T_loader, criterion: _criterion_type, save_dir: str, max_epoch: int = 100,
                 num_batches: int = 100, device="cpu", config: Dict[str, Any],
                 **kwargs) -> None:
        super().__init__(model=model, labeled_loader=labeled_loader,
                         val_loader=val_loader, test_loader=test_loader, criterion=criterion, save_dir=save_dir,
                         max_epoch=max_epoch, num_batches=num_batches, device=device, disable_bn=False,
                         two_stage=False, config=config, **kwargs)

    @property
    def train_epocher(self) -> Type[EpocherBase]:
        return FineTuneEpocher
