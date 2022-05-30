# this is the hook for the trainer.
import weakref
from contextlib import contextmanager

from loguru import logger
from torch import nn


# if t.TYPE_CHECKING:
# from contrastyou.trainer2.base import Trainer

# _Base = Trainer
# else:
#     _Base = object


class _HookMixin(object):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hooks = nn.ModuleList()

    @contextmanager
    def register_hook(self, *hook):
        if self._initialized:
            raise RuntimeError("`register_hook must be called before `init()``")
        for h in hook:
            self._hooks.append(h)
            h.to(self.device)  # put the hook into device.
            h.trainer = weakref.proxy(self)

        logger.trace("bind TrainerHooks")

        for h in self._hooks:
            h.after_initialize()
        yield
        for h in hook:
            h.close()
        logger.trace("close TrainerHooks")
