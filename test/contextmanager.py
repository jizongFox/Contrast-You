from abc import abstractmethod, ABCMeta
from typing import Union, Dict

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from contrastyou.modules.model import Model
from contrastyou.meters2 import MeterInterface, AverageValueMeter
from contrastyou import ModelState
from contextlib import contextmanager


class _Epoch(metaclass=ABCMeta):

    @contextmanager
    def register_meters(self):
        meters: MeterInterface = MeterInterface()
        meters = self._configure_meters(meters)
        yield meters

    @abstractmethod
    def _configure_meters(self, meters: MeterInterface):
        # to be overrided to add or delete individual meters
        return meters


class TrainEpoch(_Epoch):

    def _configure_meters(self, meters: MeterInterface):
        meters.register_meter("meter1", AverageValueMeter())
        return meters

    def _run_epoch(self, *args, **kwargs):
        with self.register_meters() as meters:
            print(meters.meter_names)
            meters["meter1"].add(1)
            meters["meter1"].add(2)
        print(meters.tracking_status())

    def run_epoch(self, *args, **kwargs):
        return self._run_epoch(*args, **kwargs)


class ValEpoch(TrainEpoch):

    def _configure_meters(self, meters: MeterInterface):
        meters = super(ValEpoch, self)._configure_meters(meters)
        meters.register_meter("meter2", AverageValueMeter())
        return meters


if __name__ == '__main__':
    tepoch = TrainEpoch()
    tepoch.run_epoch()

    vepoch = ValEpoch()
    vepoch.run_epoch()
