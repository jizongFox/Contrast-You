from functools import lru_cache, partial
from typing import Protocol, Sequence, ContextManager, List, Dict

from loguru import logger
from torch import nn


class _Network(Protocol):
    encoder_names: Sequence[str]
    decoder_names: Sequence[str]
    arch_elements: Sequence[str]
    layer_dimension: Dict[str, int]

    def switch_grad(self, **kwargs) -> ContextManager:
        ...

    def switch_bn_track(self, **kwargs) -> ContextManager:
        ...

    @property
    def num_classes(self) -> int:
        return ...

    def get_channel_dim(self, name: str) -> int:
        ...

    def get_module(self, name: str) -> nn.Module:
        ...


@lru_cache()
def _arch_element2index(model: _Network):
    return {k: i for i, k in enumerate(model.arch_elements)}


@lru_cache()
def _index2arch_element(index: int, *, model: _Network):
    return dict(enumerate(model.arch_elements))[index]


@lru_cache()
def arch_order(name: str, *, model):
    return _arch_element2index(model)[name]


def sort_arch(name_list: [List[str]], reverse=False, *, model) -> List[str]:
    return sorted(name_list, key=partial(arch_order, model=model), reverse=reverse)


def _check_params(start, end, include_start, include_end, *, model: _Network):
    """
    1. raise error when start is None but include_start=False
    2. raise error when end is None but include_end=False
    3. raise error when start is larger than end when both given
    4. if start or end are given, they should in a list
    """
    if start is None and include_start is False:
        raise ValueError('include_start should be True given start=None')

    if end is None and include_end is False:
        raise ValueError('include_end should be True given end=None')

    if isinstance(start, str) and start not in model.arch_elements:
        raise ValueError(start)
    if isinstance(end, str) and end not in model.arch_elements:
        raise ValueError(end)
    if isinstance(start, str) and isinstance(end, str):
        if arch_order(start, model=model) > arch_order(end, model=model):
            raise ValueError((start, end))


def _complete_arch_start2end(start: str, end: str, include_start=True, include_end=True, *, model):
    start_index, end_index = arch_order(start, model=model), arch_order(end, model=model)
    assert start_index <= end_index, (start, end)
    all_index = list(
        range(start_index if include_start else start_index + 1, end_index + 1 if include_end else end_index)
    )
    component_list = [_index2arch_element(i, model=model) for i in all_index]
    if not component_list:
        logger.opt(depth=2).debug("component list None")
    return component_list
