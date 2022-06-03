import typing as t

import torch
from loguru import logger

from contrastyou.trainer2._utils import safe_save
from contrastyou.utils import path2Path

if t.TYPE_CHECKING:
    from .base import Trainer

    _Base = Trainer
else:
    _Base = object


class IOMixin(_Base):

    def load_state_dict_from_path(self, path: str, name="last.pth", strict=True, ) -> None:
        path_ = path2Path(path)
        assert path_.exists(), path
        if path_.is_file() and path_.suffix in (".pth", ".pt"):
            path_ = path_
        elif path_.is_dir() and (path_ / name).exists():
            path_ /= name
        else:
            raise FileNotFoundError(path_)
        state_dict = torch.load(str(path_), map_location="cpu")
        self.load_state_dict(state_dict, strict)
        logger.info(f"Successfully loaded checkpoint from {str(path_)}.")

    def save_to(self, *, save_dir: str = None, save_name: str):
        assert path2Path(save_name).suffix in (".pth", ".pt"), path2Path(save_name).suffix
        if save_dir is None:
            save_dir = self._save_dir

        save_dir_ = path2Path(save_dir)
        save_dir_.mkdir(parents=True, exist_ok=True)
        state_dict = self.state_dict()
        safe_save(state_dict, str(save_dir_ / save_name))

    def resume_from_checkpoint(self, checkpoint: t.Dict[str, t.Dict], strict=True):
        self.load_state_dict(checkpoint, strict=strict)

    def resume_from_path(self, path: str, name="last.pth", strict=True, ):
        return self.load_state_dict_from_path(str(path), name, strict)

    def load_state_dict(self, state_dict, strict):
        pass
