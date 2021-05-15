# this hook collaborates with the Epocher to provide a scalable thing.
import weakref
from typing import Iterator

from torch import nn
from torch.nn import Parameter

from contrastyou.meters import MeterInterface


class _ClassNameMeta(type):
    names = []

    def __call__(cls, *args, **kwargs):
        if "hook_name" in kwargs:
            hook_name = kwargs["hook_name"]
            if hook_name in cls.names:
                raise ValueError(hook_name)
            cls.names.append(hook_name)
        return super(_ClassNameMeta, cls).__call__(*args, **kwargs)


class TrainerHook(nn.Module, metaclass=_ClassNameMeta):
    learnable_modules = ()

    def __init__(self, hook_name: str, ):
        super().__init__()
        self._hook_name = hook_name

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for m in self.learnable_modules:
            yield from m.parameters(recurse=recurse)

    @classmethod
    def create_from_trainer(cls, trainer):
        pass


class CombineTrainerHook(TrainerHook):

    def __init__(self, *trainer_hook):
        super().__init__("")
        self._hooks = nn.ModuleList(trainer_hook)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for h in self._hooks:
            yield from h.parameters(recurse=recurse)

    def __call__(self):
        return CombineEpochHook(*[h() for h in self._hooks])


class EpocherHook:

    def __init__(self, name: str, ) -> None:
        self._name = name
        self.meters: MeterInterface

    def set_epocher(self, epocher):
        self._epocher = weakref.proxy(epocher)
        self.meters = weakref.proxy(epocher.meters)
        self.configure_meters(self.meters)

    def configure_meters(self, meters: MeterInterface):
        return meters

    def before_forward_pass(self, **kwargs):
        pass

    def after_forward_pass(self, **kwargs):
        pass

    def before_regularization(self, **kwargs):
        pass

    def after_regularization(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        pass

    def close(self):
        pass


class CombineEpochHook(EpocherHook):
    def __init__(self, *epocher_hook: EpocherHook) -> None:
        self._epocher_hook = epocher_hook

    def set_epocher(self, epocher):
        for h in self._epocher_hook:
            h.set_epocher(epocher)

    def before_forward_pass(self, **kwargs):
        for h in self._epocher_hook:
            h.before_forward_pass(**kwargs)

    def after_forward_pass(self, **kwargs):
        for h in self._epocher_hook:
            h.after_forward_pass(**kwargs)

    def before_regularization(self, **kwargs):
        for h in self._epocher_hook:
            h.before_regularization(**kwargs)

    def after_regularization(self, **kwargs):
        for h in self._epocher_hook:
            h.after_regularization(**kwargs)

    def __call__(self, **kwargs):
        return sum([h(**kwargs) for h in self._epocher_hook])

    def close(self):
        for h in self._epocher_hook:
            h.close()
