from ._pretrain_helper import _FreezeGradMixin, _PretrainEpocherMixin
from .comparable import InfoNCEEpocher
from .miepocher import MITrainEpocher, ConsistencyMIEpocher
from .newepocher import ProposedEpocher1, ProposedEpocher2


class InfoNCEPretrainEpocher(_FreezeGradMixin, _PretrainEpocherMixin, InfoNCEEpocher):
    pass


class MIPretrainEpocher(_FreezeGradMixin, _PretrainEpocherMixin, MITrainEpocher):
    pass


class UDAIICPretrainEpocher(_FreezeGradMixin, _PretrainEpocherMixin, ConsistencyMIEpocher):
    pass


class ExperimentalPretrainEpocher(_FreezeGradMixin, _PretrainEpocherMixin, ProposedEpocher1):
    pass


class ExperimentalPretrainEpocher2(_FreezeGradMixin, _PretrainEpocherMixin, ProposedEpocher2):
    pass
