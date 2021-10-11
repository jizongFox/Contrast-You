from typing import Type

from contrastyou.losses.multicore_loss import GeneralOverSegmentedLoss
from ..trainer import SemiTrainer
from ...epochers.features import MultiCoreEvalEpocher, MultiCoreTrainEpocher

__all__ = ["MulticoreTrainer"]


class MulticoreTrainer(SemiTrainer):
    _criterion: GeneralOverSegmentedLoss

    def _create_initialized_eval_epoch(self, *, model, loader, **kwargs) -> MultiCoreEvalEpocher:
        epocher = MultiCoreEvalEpocher(model=model, loader=loader, sup_criterion=self._criterion,
                                       cur_epoch=self._cur_epoch,
                                       device=self._device, scaler=self.scaler, accumulate_iter=self._accumulate_iter)
        epocher.init()
        return epocher

    @property
    def train_epocher(self) -> Type[MultiCoreTrainEpocher]:
        return MultiCoreTrainEpocher

    def _init_optimizer(self):
        optimizer = super(MulticoreTrainer, self)._init_optimizer()

        optim_params = self._config["Optim"]

        optimizer.add_param_group(
            {"params": self._criterion.parameters(),
             **{k: v for k, v in optim_params.items() if k != "name" and k != "pre_lr" and k != "ft_lr"}})

        return optimizer
