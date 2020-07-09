from abc import abstractmethod, ABCMeta
from contextlib import contextmanager
from typing import Union, Dict, List

import torch

from contrastyou.meters2 import MeterInterface
from contrastyou.modules.model import Model
from ..callbacks._callback import _EpochCallack

_REPORT_DICT = Dict[str, Union[float, int]]
REPORT_DICT = Union[_REPORT_DICT, Dict[str, _REPORT_DICT]]


class _EpochMixin:

    def __init__(self, *args, **kwargs) -> None:
        super(_EpochMixin, self).__init__(*args, **kwargs)
        self._callbacks: List[_EpochCallack] = []

    def register_callbacks(self, callbacks: List[_EpochCallack]):
        if not isinstance(callbacks, list):
            callbacks = [callbacks, ]
        for i, c in enumerate(callbacks):
            if not isinstance(c, _EpochCallack):
                raise TypeError(f"callbacks [{i}] should be an instance of {_EpochCallack.__name__}, "
                                f"given {c.__class__.__name__}.")
            c.set_epocher(self)
            self._callbacks.append(c)

    def run(self, *args, **kwargs) -> REPORT_DICT:
        with self._register_meters() as self.meters:
            self._before_run()
            result = self._run(*args, **kwargs)
            self._after_run(report_dict=result)
        return result

    def step(self, *args, **kwargs) -> REPORT_DICT:
        # return accumulated dict by the design
        self._before_step()
        result = self._step(*args, **kwargs)
        self._after_step(report_dict=result)
        return result

    def _before_run(self):
        for c in self._callbacks:
            c.before_run()

    def _after_run(self, *args, **kwargs):
        for c in self._callbacks:
            c.after_run(*args, **kwargs)

    def _before_step(self):
        for c in self._callbacks:
            c.before_step()

    def _after_step(self, *args, **kwargs):
        for c in self._callbacks:
            c.after_step(*args, **kwargs)


class _Epoch(metaclass=ABCMeta):

    def __init__(self, model: Model, cur_epoch=0, device="cpu") -> None:
        super().__init__()
        self._model = model
        self._device = device
        self._cur_epoch = cur_epoch
        self.to(self._device)

    @classmethod
    def create_from_trainer(cls, trainer, *args, **kwargs):
        model = trainer._model
        device = trainer._device
        cur_epoch = trainer._cur_epoch
        return cls(model, cur_epoch, device)

    @contextmanager
    def _register_meters(self):
        meters: MeterInterface = MeterInterface()
        meters = self._configure_meters(meters)
        yield meters

    @abstractmethod
    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        # todo: to be overrided to add or delete individual meters
        return meters

    @abstractmethod
    def _run(self, *args, **kwargs) -> REPORT_DICT:
        pass

    def run(self, *args, **kwargs) -> REPORT_DICT:
        with self._register_meters() as self.meters:
            return self._run(*args, **kwargs)

    @abstractmethod
    def _step(self, *args, **kwargs) -> REPORT_DICT:
        # return accumulated dict by the design
        pass

    def step(self, *args, **kwargs) -> REPORT_DICT:
        # return accumulated dict by the design
        return self._step(*args, **kwargs)

    @abstractmethod
    def _prepare_batch(self, *args, **kwargs):
        pass

    def to(self, device="cpu"):
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self._model.to(device)
        self._device = device

    def assert_model_status(self):
        assert self._model.training, self._model.training


class Epoch(_EpochMixin, _Epoch):
    """Epocher with Mixin"""

    @classmethod
    @abstractmethod
    def create_from_trainer(cls, trainer, loader=None):
        pass
