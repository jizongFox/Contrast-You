from copy import deepcopy
from typing import Type

from deepclustering2.meters2 import EpochResultDict
from semi_seg.epochers.base import TrainEpocher
from semi_seg.epochers.newepocher import ProposedEpocher1
from semi_seg.trainers.trainer import InfoNCETrainer


class ExperimentalTrainer(InfoNCETrainer):
    def _init(self):
        super(ExperimentalTrainer, self)._init()
        config = deepcopy(self._config["NeighParams"])

        # neigh params:
        self._k = int(config["k"])
        self._m = int(config["m"])
        self._neigh_weight = float(config["neigh_weight"])

    def _set_epocher_class(self, epocher_class: Type[TrainEpocher] = ProposedEpocher1):
        super(ExperimentalTrainer, self)._set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: ProposedEpocher1, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, projectors_wrapper=self._projector,
                     infoNCE_criterion=self._criterion, kernel_size=self._k, margin=self._m,
                     neigh_weight=self._neigh_weight)
        epocher.set_global_contrast_method(method_name=self.__encoder_method__)
        result = epocher.run()
        return result


class ExperimentalTrainer2(ExperimentalTrainer):
    def _init(self):
        super(ExperimentalTrainer2, self)._init()
        from contrastyou.losses.contrast_loss import SupConLoss3 as SupConLoss
        config = deepcopy(self._config["InfoNCEParameters"])
        self._criterion = SupConLoss(temperature=config["LossParams"]["temperature"], out_mode=True)
