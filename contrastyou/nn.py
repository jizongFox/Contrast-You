from collections import OrderedDict, namedtuple
from typing import Any, Union, List, Set

import torch
from torch import nn
from torch.nn.modules.module import _addindent  # noqa
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler  # noqa

__all__ = ["ModuleBase", "Buffer", "NoTrackable"]


class ModuleBase(nn.Module):
    """
    This module is to enhance nn.Module with `NoTrackable` argument and "Buffer" argument
    """

    def __init__(self) -> None:
        super().__init__()
        self._persist_buffer: OrderedDict = OrderedDict()
        self._non_trackable_buffer: Set[str, bytes] = set()

    def __setattr__(self, name, value):
        # todo: may have bugs here
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        if isinstance(value, Buffer):
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set,
                        self._persist_buffer, self._non_trackable_buffer)
            self.register_persist_buffer(name, value.data)
        elif hasattr(self, "_persis_buffer") and name in self._persist_buffer:
            self._persist_buffer[name] = value
        elif isinstance(value, NoTrackable):
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set,
                        self._persist_buffer, self._non_trackable_buffer)
            self.register_non_trackable_buffer(name, value.data)
        elif hasattr(self, "_non_trackable_buffer") and name in self._non_trackable_buffer:
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, item):
        if "_persist_buffer" in self.__dict__ and item in self._persist_buffer:
            return self._persist_buffer[item]
        else:
            return super().__getattr__(item)

    def __delattr__(self, item):
        if item in self._non_trackable_buffer:
            self._non_trackable_buffer.remove(item)
            object.__delattr__(self, item)
        elif item in self._persist_buffer:
            del self._persist_buffer[item]
        else:
            return super(ModuleBase, self).__delattr__(item)

    def register_persist_buffer(self, name: str, data: Any):

        if '_persist_buffer' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before _TrainerBase.__init__() call")
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError(f"buffer name should be a string. Got {torch.typename(name)}")
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif not name:
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._persist_buffers:
            raise KeyError(f"attribute '{name}' already exists")
        self._persist_buffer[name] = data

    def register_non_trackable_buffer(self, name, module: nn.Module):
        if '_non_trackable_buffer' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before _TrainerBase.__init__() call")
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError(f"buffer name should be a string. Got {torch.typename(name)}")
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._non_trackable_buffer:
            raise KeyError(f"attribute '{name}' already exists")

        if name not in self._non_persistent_buffers_set:
            self._non_trackable_buffer.add(name)
        return object.__setattr__(self, name, module)

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

    def _other_state_dict(self):
        return {
            k: v.state_dict() for k, v in self.__dict__.items() if
            hasattr(v, 'state_dict') and callable(v.state_dict) and k not in self._non_trackable_buffer
        }

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        super_state = super().state_dict(destination, prefix, keep_vars)
        buffer_state = self._persist_buffer.copy()
        other_state = self._other_state_dict()

        return OrderedDict({
            "module_state": super_state,
            "buffer_state": buffer_state,
            "other_state": other_state
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

        other_dict = state_dict["other_state"]
        missing_keys.extend(list(set(self._other_state_dict()) - set(other_dict)))
        unexpected_keys.extend(list(set(other_dict) - set(self._other_state_dict())))
        for name in self._other_state_dict():
            if name in other_dict:
                getattr(self, name).load_state_dict(other_dict[name])

        if strict:
            record_err_msg(missing_keys, unexpected_keys, error_msgs)
        if error_msgs:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, "\n\t".join(error_msgs)))

        return _IncompatibleKeys(incompatible_keys.missing_keys + missing_keys,
                                 incompatible_keys.unexpected_keys + unexpected_keys)

    def to(self, device: Union[str, torch.device], **kwargs):
        for k, module in self.__dict__.items():
            if k in self._non_trackable_buffer:
                continue
            if isinstance(module, Optimizer):
                optimizer_to(module, device)
            elif isinstance(module, _LRScheduler):
                scheduler_to(module, device)

        return super().to(device=device, **kwargs)


class Buffer:
    """
    A buffer that can be used to store the state of a module.
    """

    def __init__(self, data=None):
        if isinstance(data, torch.nn.Module):
            raise ValueError(f"cannot wrap a Module in a Buffer, given {data.__class__.__name__}")

        if isinstance(data, torch.optim.Optimizer):
            raise ValueError(f"cannot wrap an Optimizer in a Buffer, given {data.__class__.__name__}")

        if isinstance(data, torch.optim.lr_scheduler._LRScheduler):  # noqa
            raise ValueError(f"cannot wrap a Scheduler in a Buffer, given {data.__class__.__name__}")

        self.data = data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"


class NoTrackable:

    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"


def record_err_msg(missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]):
    if unexpected_keys:
        error_msgs.insert(0, 'Unexpected key(s) in state_dict: {}. '.format(
            ', '.join(f'"{k}"' for k in unexpected_keys)))

    if missing_keys:
        error_msgs.insert(0, 'Missing key(s) in state_dict: {}. '.format(
            ', '.join(f'"{k}"' for k in missing_keys)))


class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__


def optimizer_to(optimizer: Optimizer, device):
    for param in optimizer.state.values():
        # Not sure if there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def scheduler_to(scheduler: _LRScheduler, device):
    for param in scheduler.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
