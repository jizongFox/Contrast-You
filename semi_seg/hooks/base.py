# this hook collaborates with the Epocher to provide a scalable thing.
from typing import Iterator

from torch import nn
from torch.nn import Parameter


class HooKBase(nn.Module):

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """this is to be called before initialization of the optimizer"""

    def configure_meters(self, meters):
        pass
