from contrastyou.datasets._seg_datset import ContrastBatchSampler  # noqa

from ._helper import _PretrainTrainerMixin
from .proposedtrainer import ExperimentalTrainer
from .trainer import InfoNCETrainer, IICTrainer, UDAIICTrainer


class PretrainInfoNCETrainer(_PretrainTrainerMixin, InfoNCETrainer):
    from semi_seg.epochers.pretrain import InfoNCEPretrainEpocher

    def _set_epocher_class(self, epocher_class=InfoNCEPretrainEpocher):
        super(PretrainInfoNCETrainer, self)._set_epocher_class(epocher_class)


class PretrainIICTrainer(_PretrainTrainerMixin, IICTrainer):
    from semi_seg.epochers.pretrain import MIPretrainEpocher

    def _set_epocher_class(self, epocher_class=MIPretrainEpocher):
        super(PretrainIICTrainer, self)._set_epocher_class(epocher_class)


class PretrainUDAIICTrainer(_PretrainTrainerMixin, UDAIICTrainer):
    from semi_seg.epochers.pretrain import UDAIICPretrainEpocher

    def _set_epocher_class(self, epocher_class=UDAIICPretrainEpocher):
        super(PretrainUDAIICTrainer, self)._set_epocher_class(epocher_class)


class PretrainExperimentalTrainer(_PretrainTrainerMixin, ExperimentalTrainer):
    from ..epochers.pretrain import ExperimentalPretrainEpocher

    def _set_epocher_class(self, epocher_class=ExperimentalPretrainEpocher):
        super(PretrainExperimentalTrainer, self)._set_epocher_class(epocher_class)
