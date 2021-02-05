from ._pretrain_helper import _FreezeGradMixin, _PretrainEpocherMixin
from .comparable import InfoNCEEpocher
from .miepocher import MITrainEpocher, ConsistencyMIEpocher
from .newepocher import NewEpocher, NewEpocher2


class InfoNCEPretrainEpocher(_FreezeGradMixin, _PretrainEpocherMixin, InfoNCEEpocher):
    pass


class MIPretrainEpocher(_FreezeGradMixin, _PretrainEpocherMixin, MITrainEpocher):
    pass


class UDAIICPretrainEpocher(_FreezeGradMixin, _PretrainEpocherMixin, ConsistencyMIEpocher):
    pass


class ExperimentalPretrainEpocher(_FreezeGradMixin, _PretrainEpocherMixin, NewEpocher):
    pass


class ExperimentalPretrainEpocher2(_FreezeGradMixin, _PretrainEpocherMixin, NewEpocher2):
    pass
