from typing import Type

from ..trainer import SemiTrainer
from ...epochers.features import MultiCoreEvalEpocher, MultiCoreTrainEpocher

__all__ = ["MulticoreTrainer"]


class MulticoreTrainer(SemiTrainer):
    def _create_initialized_eval_epoch(self, *, model, loader, **kwargs) -> MultiCoreEvalEpocher:
        epocher = MultiCoreEvalEpocher(model=model, loader=loader, sup_criterion=self._criterion,
                                       cur_epoch=self._cur_epoch,
                                       device=self._device, scaler=self.scaler, accumulate_iter=self._accumulate_iter)
        epocher.init()
        return epocher

    @property
    def train_epocher(self) -> Type[MultiCoreTrainEpocher]:
        return MultiCoreTrainEpocher
