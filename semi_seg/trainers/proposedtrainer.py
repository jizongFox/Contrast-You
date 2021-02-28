from typing import Type

from deepclustering2.meters2 import EpochResultDict
from semi_seg.epochers.base import TrainEpocher
from semi_seg.epochers.newepocher import EncoderDenseContrastEpocher, EncoderDenseMixupContrastEpocher
from semi_seg.trainers.trainer import InfoNCETrainer


class ExperimentalTrainer(InfoNCETrainer):

    def _set_epocher_class(self, epocher_class: Type[TrainEpocher] = EncoderDenseContrastEpocher):
        super(ExperimentalTrainer, self)._set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: EncoderDenseContrastEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, projectors_wrapper=self._projector,
                     infoNCE_criterion=self._criterion)
        epocher.set_global_contrast_method(method_name=self.__encoder_method__)
        result = epocher.run()
        return result


class ExperimentalTrainerwithMixUp(ExperimentalTrainer):

    def _set_epocher_class(self, epocher_class: Type[TrainEpocher] = EncoderDenseMixupContrastEpocher):
        super(ExperimentalTrainer, self)._set_epocher_class(epocher_class)
