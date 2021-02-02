from .comparable import InfoNCEEpocher
from .miepocher import MITrainEpocher, ConsistencyMIEpocher
from .newepocher import NewEpocher, NewEpocher2
from ._pretrain_helper import __FreezeGradMixin, _PretrainEpocherMixin


class InfoNCEPretrainEpocher(__FreezeGradMixin, _PretrainEpocherMixin, InfoNCEEpocher):
    pass


class MIPretrainEpocher(__FreezeGradMixin, _PretrainEpocherMixin, MITrainEpocher):
    pass


class UDAIICPretrainEpocher(__FreezeGradMixin, _PretrainEpocherMixin, ConsistencyMIEpocher):
    pass


class ExperimentalPretrainEpocher(__FreezeGradMixin, _PretrainEpocherMixin, NewEpocher):
    pass


class ExperimentalPretrainEpocher2(__FreezeGradMixin, _PretrainEpocherMixin, NewEpocher2):
    pass
