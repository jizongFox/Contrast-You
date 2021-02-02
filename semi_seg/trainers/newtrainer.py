from copy import deepcopy
from typing import Type

from deepclustering2.meters2 import EpochResultDict
from semi_seg.epochers.base import TrainEpocher
from semi_seg.epochers.newepocher import NewEpocher
from semi_seg.trainers.trainer import InfoNCETrainer, IICTrainer
from .._utils import ContrastiveProjectorWrapper


class ExperimentalTrainer(InfoNCETrainer):
    def _init(self):
        super(IICTrainer, self)._init()
        config = deepcopy(self._config["InfoNCEParameters"])
        self._projector = ContrastiveProjectorWrapper()
        self._projector.init_encoder(
            feature_names=self.feature_positions,
            **config["EncoderParams"]
        )
        self._projector.init_decoder(
            feature_names=self.feature_positions,
            **config["DecoderParams"]
        )
        from contrastyou.losses.contrast_loss import SupConLoss2 as SupConLoss

        self._criterion = SupConLoss(temperature=config["LossParams"]["temperature"], out_mode=True)
        self._reg_weight = float(config["weight"])
        self._neigh_weight = float(config["neigh_weight"])

        # neigh params:
        self._k = config["NeighParams"]["kernel_size"]
        self._m = config["NeighParams"]["margin"]

    def _set_epocher_class(self, epocher_class: Type[TrainEpocher] = NewEpocher):
        super(ExperimentalTrainer, self)._set_epocher_class(epocher_class)

    def _run_epoch(self, epocher: NewEpocher, *args, **kwargs) -> EpochResultDict:
        epocher.init(reg_weight=self._reg_weight, projectors_wrapper=self._projector,
                     infoNCE_criterion=self._criterion, kernel_size=self._k, margin=self._m,
                     neigh_weight=self._neigh_weight)
        result = epocher.run()
        return result


class ExperimentalTrainer2(ExperimentalTrainer):
    def _init(self):
        super(ExperimentalTrainer2, self)._init()
        from contrastyou.losses.contrast_loss import SupConLoss3 as SupConLoss
        config = deepcopy(self._config["InfoNCEParameters"])
        self._criterion = SupConLoss(temperature=config["LossParams"]["temperature"], out_mode=True)
