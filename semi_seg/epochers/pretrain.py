from ._mixins import _PretrainEpocherMixin
from .comparable import InfoNCEEpocher
from .miepocher import MITrainEpocher, ConsistencyMIEpocher
from .newepocher import EncoderDenseContrastEpocher, EncoderDenseMixupContrastEpocher, \
    EncoderMultiTaskContrastiveEpocher


# override batch loop in order to ignore the supervised loss.
class InfoNCEPretrainEpocher(_PretrainEpocherMixin, InfoNCEEpocher):
    pass


class MIPretrainEpocher(_PretrainEpocherMixin, MITrainEpocher):
    pass


class UDAIICPretrainEpocher(_PretrainEpocherMixin, ConsistencyMIEpocher):
    pass


class ExperimentalPretrainEpocher(_PretrainEpocherMixin, EncoderDenseContrastEpocher):
    pass


class ExperimentalPretrainMixinEpocher(_PretrainEpocherMixin, EncoderDenseMixupContrastEpocher):
    pass


class ExperimentPretrainMultiTaskEpocher(_PretrainEpocherMixin, EncoderMultiTaskContrastiveEpocher):
    pass
