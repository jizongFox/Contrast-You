from typing import List

from torch import nn

from contrastyou.arch.unet import freeze_grad
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation  # noqa
from contrastyou.epocher._utils import write_predict, write_img_target  # noqa
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.type import T_loss
from .base import PretrainEpocher
from .comparable import InfoNCEEpocher
from .miepocher import IICTrainEpocher, UDAIICEpocher
from .._utils import ContrastiveProjectorWrapper, ClusterProjectorWrapper, IICLossWrapper


class _freeze_grad_mixin:
    _model: nn.Module
    _feature_position: List[str]

    def _run(self, *args, **kwargs):
        with freeze_grad(self._model, self._feature_position) as self._model:  # noqa
            return super()._run(*args, **kwargs)


class InfoNCEPretrainEpocher(_freeze_grad_mixin, PretrainEpocher, InfoNCEEpocher):

    def init(self, *, chain_dataloader=None, projectors_wrapper: ContrastiveProjectorWrapper = None,
             infoNCE_criterion: T_loss = None, **kwargs):
        super(InfoNCEPretrainEpocher, self).init(projectors_wrapper=projectors_wrapper,
                                                 infoNCE_criterion=infoNCE_criterion, chain_dataloader=chain_dataloader)


class IICPretrainEpocher(_freeze_grad_mixin, PretrainEpocher, IICTrainEpocher):

    def init(self, *, chain_dataloader=None, projectors_wrapper: ClusterProjectorWrapper = None,
             IIDSegCriterionWrapper: IICLossWrapper = None, enforce_matching=False, **kwargs):
        super().init(chain_dataloader=chain_dataloader, reg_weight=1.0, projectors_wrapper=projectors_wrapper,
                     IIDSegCriterionWrapper=IIDSegCriterionWrapper, enforce_matching=enforce_matching, **kwargs)


class UDAIICPretrainEpocher(_freeze_grad_mixin, PretrainEpocher, UDAIICEpocher):
    def init(self, *, chain_dataloader=None, iic_weight: float = None, uda_weight: float = None,
             projectors_wrapper: ClusterProjectorWrapper = None, IIDSegCriterionWrapper: IICLossWrapper = None,
             reg_criterion: T_loss = None, enforce_matching=False, **kwargs):
        super().init(chain_dataloader=chain_dataloader, iic_weight=iic_weight, uda_weight=uda_weight,
                     projectors_wrapper=projectors_wrapper,
                     IIDSegCriterionWrapper=IIDSegCriterionWrapper, reg_criterion=reg_criterion,
                     enforce_matching=enforce_matching, **kwargs)
