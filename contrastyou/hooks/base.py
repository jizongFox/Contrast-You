# this hook collaborates with the Epocher to provide a scalable thing.
import weakref
from typing import Iterator, List

from torch import nn
from torch.nn import Parameter

from contrastyou.meters import MeterInterface
from contrastyou.utils import class_name


class HookNameExistError(Exception):
    pass


class _ClassNameMeta(type):
    names: List[str] = []

    def __call__(cls, *args, **kwargs):
        if "hook_name" in kwargs:
            hook_name = kwargs["hook_name"]
            if hook_name in cls.names:
                raise HookNameExistError(hook_name)
            cls.names.append(hook_name)
        return super(_ClassNameMeta, cls).__call__(*args, **kwargs)


class TrainerHook(nn.Module, metaclass=_ClassNameMeta):

    def __init__(self, *, hook_name: str):
        super().__init__()
        self._hook_name = hook_name

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for m in self.learnable_modules:
            yield from m.parameters(recurse=recurse)

    @property
    def learnable_modules(self) -> List[nn.Module]:
        return []

    def __call__(self, **kwargs):
        raise NotImplementedError(f"subclass {class_name(self)} must implement __call__ function.")

    def close(self):
        pass


class CombineTrainerHook(TrainerHook):

    def __init__(self, *trainer_hook: TrainerHook):
        super().__init__(hook_name="")
        self._hooks = nn.ModuleList(trainer_hook)

    def __call__(self):
        return CombineEpochHook(*[h() for h in self._hooks])

    @property
    def learnable_modules(self):
        return self._hooks

    def close(self):
        for h in self._hooks:
            h.close()


class EpocherHook:

    def __init__(self, *, name: str) -> None:
        self._name = name

    def set_epocher(self, epocher):
        self._epocher = weakref.proxy(epocher)
        self.meters = weakref.proxy(epocher.meters)
        self.configure_meters(self.meters)

    @property
    def epocher(self):
        return self._epocher

    def configure_meters(self, meters: MeterInterface):
        return meters

    def before_batch_update(self, **kwargs):
        pass

    def before_forward_pass(self, **kwargs):
        pass

    def after_forward_pass(self, **kwargs):
        pass

    def before_regularization(self, **kwargs):
        pass

    def after_regularization(self, **kwargs):
        pass

    def after_batch_update(self, **kwargs):
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

    def before_batch_update(self, **kwargs):
        for h in self._epocher_hook:
            h.before_batch_update(**kwargs)

    def after_batch_update(self, **kwargs):
        for h in self._epocher_hook:
            h.after_batch_update(**kwargs)

    def __call__(self, **kwargs):
        return sum([h(**kwargs) for h in self._epocher_hook])

    def close(self):
        for h in self._epocher_hook:
            h.close()
