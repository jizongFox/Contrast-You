# this is the hook for the trainer.
import typing as t
from contextlib import contextmanager

from loguru import logger
from torch import nn

# _Base = Trainer
# else:
#     _Base = object
from contrastyou.hooks import TrainerHook

if t.TYPE_CHECKING:
    from contrastyou.trainer.base import Trainer

    _Base = Trainer
else:
    _Base = object


class HookMixin(_Base):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hooks = nn.ModuleList()

    @contextmanager
    def register_hook(self, *hook: "TrainerHook"):
        if self._initialized:
            raise RuntimeError("`register_hook must be called before `init()``")
        for h in hook:
            self._hooks.append(h)
            h.to(self.device)  # put the hook into device.
            h.register_non_trackable_buffer("trainer", self)
            if hasattr(h, "_trainer"):
                del h._trainer
            h.register_non_trackable_buffer("_trainer", self)

        logger.trace("bind TrainerHooks")

        for h in self._hooks:
            h.after_initialize()
        yield
        for h in hook:
            h.close()
        logger.trace("close TrainerHooks")
