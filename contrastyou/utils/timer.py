import functools
import time
from abc import ABCMeta

import torch
from loguru import logger

from contrastyou.meters import AverageValueMeter


class EpocherTimer(metaclass=ABCMeta):

    def __init__(self) -> None:
        super().__init__()
        self._batch_time = AverageValueMeter()
        self._data_fetch_time = AverageValueMeter()
        self.__batch_end = time.time()
        self.__batch_start = time.time()

    def record_batch_start(self):
        self.__batch_start = time.time()
        self._data_fetch_time.add(self.__batch_start - self.__batch_end)

    def record_batch_end(self):
        previous_batch_end = self.__batch_end

        self.__batch_end = time.time()
        self._batch_time.add(self.__batch_end - previous_batch_end)

    def summary(self):
        return {"batch_time": self._batch_time.summary()["mean"], "fetch_time": self._data_fetch_time.summary()["mean"]}

    def __enter__(self):
        self.record_batch_start()
        return self

    def __exit__(self, *args, **kwargs):
        self.record_batch_end()


class gpu_timeit:

    def __init__(self, message="") -> None:
        super().__init__()
        self.message = message

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        elapsed_time = self.start.elapsed_time(self.end)
        logger.opt(depth=1).info(f"{self.message} operation time: {elapsed_time / 1000:.4f}.")

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapped_func
