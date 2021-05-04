# this hook collaborates with the Trainer to provide a scalable thing.
from typing import Iterator

from torch import nn
from torch.nn import Parameter


class HooKBase(nn.Module):
    @classmethod
    def create_from_trainer(cls, trainer):
        pass

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """this is to be called before initialization of the optimizer"""
        pass

    def bind_epocher(self):
        pass

    def configure_meters(self, meters):
        pass
