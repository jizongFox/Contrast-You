from collections import OrderedDict
from typing import Any, Union, List

import torch
from torch import nn
from torch.nn.modules.module import _addindent
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler  # noqa

from contrastyou.trainer._buffer import Buffer, _IncompatibleKeys
from contrastyou.trainer._functional import optimizer_to, scheduler_to


def record_err_msg(missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]):
    if unexpected_keys:
        error_msgs.insert(0, 'Unexpected key(s) in state_dict: {}. '.format(
            ', '.join(f'"{k}"' for k in unexpected_keys)))

    if missing_keys:
        error_msgs.insert(0, 'Missing key(s) in state_dict: {}. '.format(
            ', '.join(f'"{k}"' for k in missing_keys)))


class TrainerBase(nn.Module):

    def __init__(self, *, model: nn.Module, criterion, tra_loader, val_loader, save_dir, max_epoch:
    int, num_batches: int, config, **kwargs) -> None:
        super().__init__()
        self._persist_buffer: OrderedDict = OrderedDict()
        self._model = model
        self._criterion = criterion
        self._tra_loader = tra_loader
        self._val_loader = val_loader
        self._save_dir = Buffer(save_dir)
        self._max_epoch = Buffer(max_epoch)
        self._num_batches = Buffer(num_batches)
        self._config = Buffer(config)

    def __setattr__(self, key, value):
        if isinstance(value, Buffer):
            self._persist_buffer[key] = value.data
        else:
            super().__setattr__(key, value)

    def __getattr__(self, item):
        if item in self._persist_buffer:
            return self._persist_buffer[item]
        else:
            return super().__getattr__(item)

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        if extra_repr := self.extra_repr():
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append(f'({key}): {mod_str}')

        for key, module in self._persist_buffer.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append(f'({key}): {mod_str}')
        lines = extra_lines + child_lines

        main_str = f'{self._get_name()}('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def _optimizer_state_dict(self):
        return {k: v.state_dict() for k, v in self.__dict__.items() if isinstance(v, Optimizer)}

    def _scheduler_state_dict(self):
        return {k: v.state_dict() for k, v in self.__dict__.items() if isinstance(v, _LRScheduler)}

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        super_state = super().state_dict(destination, prefix, keep_vars)
        buffer_state = self._persist_buffer.copy()
        optimizer_state = self._optimizer_state_dict()
        scheduler_state = self._scheduler_state_dict()
        return OrderedDict({
            "module_state": super_state,
            "buffer_state": buffer_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state
        })

    def load_state_dict(self, state_dict: OrderedDict[str, Any], strict=True):
        if "module_state" not in state_dict:
            raise ValueError("Missing module_state in state_dict")
        incompatible_keys = super().load_state_dict(state_dict["module_state"], strict)

        error_msgs = []
        buffer_dict = state_dict["buffer_state"]
        missing_keys = list(set(self._persist_buffer.keys()) - set(buffer_dict.keys()))
        unexpected_keys = list(set(buffer_dict.keys()) - set(self._persist_buffer.keys()))

        for key in self._persist_buffer.keys():
            if key in buffer_dict:
                self._persist_buffer[key] = buffer_dict[key]

        optimizer_dict = state_dict["optimizer_state"]
        missing_keys.extend(list(set(self._optimizer_state_dict()) - set(optimizer_dict)))
        unexpected_keys.extend(list(set(optimizer_dict) - set(self._optimizer_state_dict())))

        for name in self._optimizer_state_dict():
            if name in optimizer_dict:
                getattr(self, name).load_state_dict(optimizer_dict[name], )

        scheduler_dict = state_dict["scheduler_state"]
        missing_keys.extend(list(set(self._scheduler_state_dict()) - set(scheduler_dict)))
        unexpected_keys.extend(list(set(scheduler_dict) - set(self._scheduler_state_dict())))

        for name in self._scheduler_state_dict():
            if name in scheduler_dict:
                getattr(self, name).load_state_dict(scheduler_dict[name])

        if strict:
            record_err_msg(missing_keys, unexpected_keys, error_msgs)
        if error_msgs:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, "\n\t".join(error_msgs)))

        return _IncompatibleKeys(incompatible_keys.missing_keys + missing_keys,
                                 incompatible_keys.unexpected_keys + unexpected_keys)

    def to(self, device: Union[str, torch.device], **kwargs):

        for k, module in self.__dict__.items():
            if isinstance(module, Optimizer):
                optimizer_to(module, device)
            elif isinstance(module, _LRScheduler):
                scheduler_to(module, device)

        return super().to(device=device, **kwargs)


if __name__ == '__main__':
    a = TrainerBase(model=nn.Linear(1, 1), criterion=nn.MSELoss(), tra_loader=None, val_loader=None, save_dir=None,
                    max_epoch=1, num_batches=1, config=None, extra_opt=True)
    b = TrainerBase(model=nn.Linear(1, 1), criterion=nn.MSELoss(), tra_loader=None, val_loader=None, save_dir="dfadfsa",
                    max_epoch=1, num_batches=10, config=1, )
    b.fds = Buffer(2)
    a.dfsfs = Buffer(3)
    print(a._num_batches)
    print(a.load_state_dict(b.state_dict(), strict=True))
    print(a)
    # print(str(b._optimizer))
