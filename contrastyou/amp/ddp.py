from functools import lru_cache
from typing import Optional

import torch.distributed as dist
from torch import nn


def convert2syncBN(network: nn.Module):
    return nn.SyncBatchNorm.convert_sync_batchnorm(network)


def disable_output():
    def print_pass(*args):
        pass

    import builtins

    builtins.print = print_pass


class DDPMixin:
    @property
    @lru_cache()
    def rank(self) -> Optional[int]:
        try:
            return dist.get_rank()
        except (AssertionError, AttributeError, RuntimeError):
            return None

    @property
    @lru_cache()
    def on_master(self) -> bool:
        return (self.rank == 0) or (self.rank is None)
