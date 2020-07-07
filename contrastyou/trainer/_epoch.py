from abc import abstractmethod
from typing import Union, Dict

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from contrastyou.modules.model import Model
from contrastyou.meters2 import MeterInterface
from contrastyou import ModelState
from contextlib import contextmanager
class _Epoch:

    @abstractmethod
    def register_meters(self):
        self._meters: MeterInterface = MeterInterface()

    def write2tensorboard(self):
        pass

    def _data_preprocessing(self):
        pass

    def run_epoch(self):
        pass


class TrainEpoch(_Epoch):

    def __init__(self, model:Model, train_loader: Union[DataLoader, _BaseDataLoaderIter], num_batches: int = 512) -> None:
        super().__init__()
        self._model = model
        self._loader = train_loader
        self._num_batches = num_batches
        self._indicator = range(self._num_batches)

    @contextmanager
    def register_meters(self):
        super(TrainEpoch, self).register_meters()
        self._meters.register_meter()
        yield self._meters
        self._meters.reset()


    def run_epoch(self, mode=ModelState.TRAIN ) -> Dict[str, float]:
        self._model.set_mode(mode)

        with self.register_meters() as meters:
            pass





class ValEpoch(_Epoch):
    pass
