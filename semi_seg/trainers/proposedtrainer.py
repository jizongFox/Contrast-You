from typing import Type

from deepclustering2.meters2 import EpochResultDict

from semi_seg.epochers.base import TrainEpocher
from semi_seg.epochers.newepocher import EncoderDenseContrastEpocher, EncoderDenseMixupContrastEpocher
from semi_seg.trainers.trainer import InfoNCETrainer


class ExperimentalTrainer(InfoNCETrainer):

    def _set_epocher_class(self, epocher_class: Type[TrainEpocher] = EncoderDenseContrastEpocher):
        super(ExperimentalTrainer, self)._set_epocher_class(epocher_class)



class ExperimentalTrainerwithMixUp(ExperimentalTrainer):

    def _set_epocher_class(self, epocher_class: Type[TrainEpocher] = EncoderDenseMixupContrastEpocher):
        super(ExperimentalTrainer, self)._set_epocher_class(epocher_class)


