import weakref
from abc import abstractmethod, ABCMeta
from contextlib import contextmanager
from typing import Union, Optional

import torch
import torch.distributed as dist
from loguru import logger
from torch import nn

from contrastyou.meters import MeterInterface
from ..meters.averagemeter import AverageValueListMeter
from ..mytqdm import tqdm


class DDPMixin:
    @property
    def rank(self) -> Optional[int]:
        try:
            return dist.get_rank()
        except (AssertionError, AttributeError, RuntimeError):
            return None

    def on_master(self) -> bool:
        return (self.rank == 0) or (self.rank is None)


class _Epocher(DDPMixin, metaclass=ABCMeta):
    def __init__(self, *, model: nn.Module, num_batches: int, cur_epoch=0, device="cpu") -> None:
        super().__init__()
        self._model = model
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self._num_batches = num_batches
        self._cur_epoch = cur_epoch

        self.meters = MeterInterface()
        self.indicator = tqdm(range(self._num_batches))
        self.__epocher_initialized__ = False
        self.__bind_trainer_done__ = False
        self._trainer = None

    @property
    def device(self):
        return self._device

    def _init(self, **kwargs):
        pass

    def init(self, **kwargs):
        self._init(**kwargs)
        self.configure_meters(self.meters)
        self.__epocher_initialized__ = True

    @contextmanager
    def _register_indicator(self):
        assert isinstance(
            self._num_batches, int
        ), f"self._num_batches must be provided as an integer, given {self._num_batches}."

        self.indicator.set_desc_from_epocher(self)
        yield
        self.indicator.close()
        msg = self.indicator.print_result()
        logger.opt(depth=5).info(msg)

    @contextmanager
    def _register_meters(self):
        meters = self.meters
        yield meters
        meters.join()

    @abstractmethod
    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("lr", AverageValueListMeter())
        return meters

    @abstractmethod
    def _run(self, **kwargs):
        pass

    def run(self, **kwargs):
        if not self.__epocher_initialized__:
            raise RuntimeError(f"{self.__class__.__name__} should be initialized first by calling `init()`.")
        self.to(self.device)  # put all things into the same device
        with self._register_meters(), \
            self._register_indicator():
            return self._run(**kwargs)

    def get_metric(self):
        if not self.__epocher_initialized__:
            raise RuntimeError(f"{self.__class__.__name__} should be initialized by calling `init()` before.")
        return dict(self.meters.statistics())

    def get_score(self):
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

    @staticmethod
    @abstractmethod
    def _unzip_data(data, device):
        pass

    def batch_optimization(self, **kwargs):
        return self._batch_optimization(**kwargs)

    def _batch_optimization(self, **kwargs):
        pass


"""
class _EpocherWithPlugin(_Epocher):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hooks: List[EpocherHook] = []

    def register_hooks(self, hooks: Union[EpocherHook, List[EpocherHook]]):
        hooks = hooks if isinstance(hooks, (tuple, list)) else [hooks, ]
        for h in hooks:
            h.bind_epocher(self)
        self._hooks.extend(hooks)

    def hooks_before_epoch(self):
        for h in self._hooks:
            h.before_epoch()

    def hooks_end_epoch(self):
        for h in self._hooks:
            h.end_epoch()

    def hooks_before_update(self):
        for h in self._hooks:
            h.before_update()

    def hooks_end_update(self):
        for h in self._hooks:
            h.end_update()

    def _run(self, **kwargs):
        self.hooks_before_epoch()
        result = super(HookMixin, self)._run(**kwargs)  # noqa
        self.hooks_end_epoch()
        return result
"""
