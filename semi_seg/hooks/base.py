# this hook collaborates with the Epocher to provide a scalable thing.
from typing import Iterator

from deepclustering2.meters2 import MeterInterface
from torch import nn
from torch.nn import Parameter


class ClassNameMeta(type):
    names = []

    def __call__(cls, *args, **kwargs):
        if "hook_name" in kwargs:
            hook_name = kwargs["hook_name"]
            if hook_name in cls.names:
                raise ValueError(hook_name)
            cls.names.append(hook_name)
        return super(ClassNameMeta, cls).__call__(*args, **kwargs)


class TrainHook(nn.Module, metaclass=ClassNameMeta):
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


class EpochHook:

    def __init__(self, name: str, ) -> None:
        self._name = name
        self.meters = MeterInterface()

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
