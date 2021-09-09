from copy import deepcopy

from torch import nn

from contrastyou.hooks.base import TrainerHook, EpocherHook
from contrastyou.types import CriterionType


class MetaLabelTrainerHook(TrainerHook):

    def __init__(self, *, hook_name: str, model: nn.Module, meta_weight: float, meta_criterion: CriterionType):
        super().__init__(hook_name=hook_name)
        self._model = model
        self._teacher_model = deepcopy(model)
        # initialize teacher network 
        self._meta_weight = meta_weight
        self._meta_criterion = meta_criterion

    def __call__(self):
        return MetaLabelEpocherHook(...)


class MetaLabelEpocherHook(EpocherHook):
    pass
