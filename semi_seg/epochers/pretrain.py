from ._helper import _PretrainEpocherMixin
from .comparable import InfoNCEEpocher
from .miepocher import MITrainEpocher, ConsistencyMIEpocher
from .newepocher import ProposedEpocher1, ProposedEpocher2, ProposedEpocher3


# override batch loop in order to ignore the supervised loss.
class InfoNCEPretrainEpocher(_PretrainEpocherMixin, InfoNCEEpocher):
    pass


class MIPretrainEpocher(_PretrainEpocherMixin, MITrainEpocher):
    pass


class UDAIICPretrainEpocher(_PretrainEpocherMixin, ConsistencyMIEpocher):
    pass


class ExperimentalPretrainEpocher(_PretrainEpocherMixin, ProposedEpocher1):
    pass


class ExperimentalPretrainEpocher2(_PretrainEpocherMixin, ProposedEpocher2):
    pass


class ExperimentalPretrainEpocher3(_PretrainEpocherMixin, ProposedEpocher3):
    pass
