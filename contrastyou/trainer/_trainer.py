from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Union, Dict, Any, TypeVar

import numpy as np
import torch
from deepclustering.utils import path2Path
from deepclustering.writer import SummaryWriter
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from ._buffer import _BufferMixin
from .. import PROJECT_PATH
# from ..meters import MeterInterface, AverageValueMeter
from ..modules.model import Model
from ..storage import Storage

N = TypeVar('N', int, float, Tensor, np.ndarray)


class _Trainer(_BufferMixin):
    """
    Abstract class for a general trainer, which has _train_loop, _eval_loop,load_state, state_dict, and save_checkpoint
    functions. All other trainers are the subclasses of this class.
    """
    RUN_PATH = str(Path(PROJECT_PATH) / "runs")
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / "archives")
    checkpoint_identifier = "last.pth"

    def __init__(
        self,
        model: Model,
        train_loader: Union[DataLoader, _BaseDataLoaderIter],
        val_loader: DataLoader,
        max_epoch: int = 100,
        save_dir: str = "base",
        checkpoint: str = None,
        device="cpu",
        config: dict = None,
    ) -> None:
        super(_Trainer, self).__init__()
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader

        self.register_buffer("_max_epoch", int(max_epoch))
        self.register_buffer("_best_score", -1.0)
        self.register_buffer("_start_epoch", 0)  # whether 0 or loaded from the checkpoint.
        self.register_buffer("_epoch", None)

        self._save_dir: Path = Path(self.RUN_PATH) / str(save_dir)
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self._checkpoint = checkpoint
        self._device = torch.device(device)

        if config:
            self._config = deepcopy(config)
            self._config.pop("Config", None)

        self._storage = Storage()

    def to(self, device):
        self._model.to(device=device)

    def _start_training(self):
        for epoch in range(self._start_epoch, self._max_epoch):
            if self._model.get_lr() is not None:
                self._meter_interface["lr"].add(self._model.get_lr()[0])
            self.train_loop(train_loader=self._train_loader, epoch=epoch)
            with torch.no_grad():
                current_score = self.eval_loop(self._val_loader, epoch)
            self._model.schedulerStep()
            # save meters and checkpoints
            self._meter_interface.step()
            self.save_checkpoint(self.state_dict(), epoch, current_score)
            self._meter_interface.summary().to_csv(self._save_dir / "wholeMeter.csv")

    def start_training(self):
        with SummaryWriter(log_dir=self._save_dir) as self.writer:
            return self._start_training()

    @abstractmethod
    def _train_loop(
        self,
        train_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
        epoch: int = 0,
        mode=None,
        *args,
        **kwargs,
    ):
        pass

    def train_loop(self, *args, **kwargs):
        return self._train_loop(*args, **kwargs)

    @abstractmethod
    def _eval_loop(
        self,
        val_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
        epoch: int = 0,
        mode=None,
    ) -> float:
        pass

    def eval_loop(self, *args, **kwargs):
        return self._eval_loop(*args, **kwargs)

    def inference(self, identifier="best.pth", *args, **kwargs):
        """
        Inference using the checkpoint, to be override by subclasses.
        :param args:
        :param kwargs:
        :return:
        """
        if self._checkpoint is None:
            self._checkpoint = self._save_dir
        assert Path(self._checkpoint).exists(), Path(self._checkpoint)
        assert (Path(self._checkpoint).is_dir() and identifier is not None) or (
            Path(self._checkpoint).is_file() and identifier is None
        )

        state_dict = torch.load(
            str(Path(self._checkpoint) / identifier)
            if identifier is not None
            else self._checkpoint,
            map_location=torch.device("cpu"),
        )
        self.load_checkpoint(state_dict)
        self._model.to(self._device)
        # to be added
        # probably call self._eval() method.

    def state_dict(self) -> Dict[str, Any]:
        """
        return trainer's state dict. The dict is built by considering all the submodules having `state_dict` method.
        """
        buffer_state_dict = self.buffer_state_dict()
        local_modules = {k: v for k, v in self.__dict__.items() if k != "_buffers"}

        local_state_dict = {}
        for module_name, module in local_modules.items():
            if hasattr(module, "state_dict") and callable(getattr(module, "state_dict", None)):
                local_state_dict[module_name] = module.state_dict()
        destination = {**local_state_dict, **{"_buffers": buffer_state_dict}}
        return destination

    def load_state_dict(self, state_dict) -> None:
        """
        Load state_dict for submodules having "load_state_dict" method.
        :param state_dict:
        :return:
        """
        missing_keys = []
        unexpected_keys = []
        er_msgs = []

        for module_name, module in self.__dict__.items():
            if module_name == "_buffers":
                self.load_buffer_state_dict(state_dict["_buffers"])
            if hasattr(module, "load_state_dict") and callable(getattr(module, "load_state_dict", None)):
                try:
                    module.load_state_dict(state_dict[module_name])
                except KeyError:
                    missing_keys.append(module_name)
                except Exception as ex:
                    er_msgs.append(
                        "while copying {} parameters, "
                        "error {} occurs".format(module_name, ex)
                    )
        if len(er_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, "\n\t".join(er_msgs)))

    def save_checkpoint(
        self, state_dict, current_epoch, cur_score, save_dir=None, save_name=None
    ):
        """
        save checkpoint with adding 'epoch' and 'best_score' attributes
        :param state_dict:
        :param current_epoch:
        :param cur_score:
        :return:
        """
        save_best: bool = True if float(cur_score) > float(self._best_score) else False
        if save_best:
            self._best_score = float(cur_score)
        save_dir = self._save_dir if save_dir is None else path2Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if save_name is None:
            # regular saving
            torch.save(state_dict, str(save_dir / "last.pth"))
            if save_best:
                torch.save(state_dict, str(save_dir / "best.pth"))
        else:
            # periodic saving
            torch.save(state_dict, str(save_dir / save_name))

    def load_checkpoint_from_path(self, checkpoint_path):
        checkpoint_path = path2Path(checkpoint_path)
        assert checkpoint_path.exists(), checkpoint_path
        if checkpoint_path.is_dir():
            state_dict = torch.load(
                str(Path(checkpoint_path) / self.checkpoint_identifier),
                map_location=torch.device("cpu"),
            )
        else:
            assert checkpoint_path.suffix == ".pth", checkpoint_path
            state_dict = torch.load(
                str(checkpoint_path), map_location=torch.device("cpu"),
            )
        self.load_checkpoint(state_dict)

    def clean_up(self, wait_time=3):
        """
        Do not touch
        :return:
        """
        import shutil
        import time

        time.sleep(wait_time)  # to prevent that the call_draw function is not ended.
        Path(self.ARCHIVE_PATH).mkdir(exist_ok=True, parents=True)
        sub_dir = self._save_dir.relative_to(Path(self.RUN_PATH))
        save_dir = Path(self.ARCHIVE_PATH) / str(sub_dir)
        if Path(save_dir).exists():
            shutil.rmtree(save_dir, ignore_errors=True)
        shutil.move(str(self._save_dir), str(save_dir))
        shutil.rmtree(str(self._save_dir), ignore_errors=True)
