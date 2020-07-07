from collections import OrderedDict
from copy import deepcopy
from typing import Union, TypeVar

import numpy as np
import torch
from torch import Tensor

N = TypeVar('N', int, float, Tensor, np.ndarray)


class _BufferMixin:
    """
    The buffer in Trainer is for automatic loading and saving.
    """

    def __init__(self) -> None:
        self._buffers = OrderedDict()

    def register_buffer(self, name: str, value: Union[str, N]):
        r"""Adds a persistent buffer to the module.
        """
        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("buffer name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        else:
            self._buffers[name] = value

    def __getattr__(self, name):
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        buffers = self.__dict__.get('_buffers')
        if buffers is not None and name in buffers:
            buffers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._buffers:
            del self._buffers[name]
        else:
            object.__delattr__(self, name)

    def buffer_state_dict(self):
        destination = OrderedDict()
        for name, buf in self._buffers.items():
            value = buf
            if isinstance(buf, Tensor):
                value = buf.detach()
            if isinstance(buf, np.ndarray):
                value = deepcopy(buf)
            destination[name] = value
        return destination

    def _load_buffer_from_state_dict(self, state_dict, prefix, strict,
                                     missing_keys, unexpected_keys, error_msgs):

        local_name_params = self._buffers.items()
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                with torch.no_grad():
                    try:
                        if isinstance(input_param, Tensor):
                            param.copy_(input_param)
                        else:
                            self._buffers[name] = input_param
                    except Exception as ex:
                        error_msgs.append('While copying the parameter named "{}", '
                                          'an exception occured : {}.'
                                          .format(key, ex.args))
            elif strict:
                missing_keys.append(key)

    def load_buffer_state_dict(self, state_dict):
        r"""
        """
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        state_dict = state_dict.copy()

        def load(module, prefix=''):
            module._load_buffer_from_state_dict(
                state_dict, prefix, True, missing_keys, unexpected_keys, error_msgs)

        load(self)

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, "\n\t".join(error_msgs)))
        return missing_keys, unexpected_keys, error_msgs
