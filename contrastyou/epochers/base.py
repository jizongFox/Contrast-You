import weakref
from abc import abstractmethod, ABCMeta
from contextlib import contextmanager
from typing import Union, Dict, List

import torch
from torch import nn
from torch.cuda.amp import GradScaler

from ..amp import AMPScaler, DDPMixin
from ..hooks.base import EpocherHook
from ..meters import MeterInterface, AverageValueListMeter
from ..mytqdm import tqdm


class EpocherBase(AMPScaler, DDPMixin, metaclass=ABCMeta):
    """ EpocherBase class to control the behavior of the training within one epoch.
    >>> hooks = ...
    >>> epocher = EpocherBase(...)
    >>> with epocher.register_hook(*hooks):
    >>>     epocher.run()
    >>> epocher_result, best_score = epocher.get_metric(), epocher.get_score()
    """

    meter_focus = "tra"

    def __init__(self, *, model: nn.Module, num_batches: int, cur_epoch=0, device="cpu", scaler: GradScaler,
                 accumulate_iter: int, **kwargs) -> None:
        super().__init__(scaler=scaler, accumulate_iter=accumulate_iter)
        self._model = model
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self._num_batches = num_batches
        self._cur_epoch = cur_epoch

        self.meters = MeterInterface(default_focus=self.meter_focus)
        self.configure_meters(self.meters)

        self.indicator = tqdm(range(self._num_batches), disable=not self.on_master(), leave=True, ncols=2)

        self._trainer = None
        self.__bind_trainer_done__ = False
        self._hooks: List[EpocherHook] = []

    @contextmanager
    def register_hook(self, *hook: EpocherHook):
        """
        >>> epocher = EpocherBase(...)
        >>> hooks = ...
        >>> with epocher.register_hook(*hooks):
        >>>     epocher.run()
        """
        for h in hook:
            self._hooks.append(h)
            h.set_epocher(self)
        yield
        self.close_hook()

    def close_hook(self):
        for h in self._hooks:
            h.close()

    @property
    def device(self):
        return self._device

    @contextmanager
    def _register_indicator(self):
        assert isinstance(
            self._num_batches, int
        ), f"self._num_batches must be provided as an integer, given {self._num_batches}."
        self.indicator.set_desc_from_epocher(self)
        yield
        self.indicator.close()
        self.indicator.log_result()

    @abstractmethod
    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("lr", AverageValueListMeter())
        return meters

    @abstractmethod
    def _run(self, **kwargs):
        raise NotImplementedError()

    def run(self, **kwargs):
        self.to(self.device)  # put all things into the same device

        with self.meters, self._register_indicator():
            run_result = self._run(**kwargs)

        return run_result

    def get_metric(self) -> Dict[str, Dict[str, float]]:
        return dict(self.meters.statistics())

    def get_score(self) -> float:
        raise NotImplementedError()

    def to(self, device: Union[torch.device, str] = torch.device("cpu")):
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        for n, m in self.__dict__.items():
            if isinstance(m, nn.Module):
                m.to(device)
        self._device = device

    def set_trainer(self, trainer):
        self._trainer = weakref.proxy(trainer)
        self.__bind_trainer_done__ = True

    @property
    def trainer(self):
        if not self.__bind_trainer_done__:
            raise RuntimeError(f"{self.__class__.__name__} should call `set_trainer` first")
        return self._trainer
