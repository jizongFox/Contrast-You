from deepclustering2.type import T_loss
from .base import PretrainEpocher
from .comparable import InfoNCEEpocher
from .._utils import ContrastiveProjectorWrapper


class InfoNCEPretrainEpocher(PretrainEpocher, InfoNCEEpocher):

    def init(self, *, chain_dataloader=None, projectors_wrapper: ContrastiveProjectorWrapper = None,
             infoNCE_criterion: T_loss = None, **kwargs):
        super(InfoNCEPretrainEpocher, self).init(projectors_wrapper=projectors_wrapper,
                                                 infoNCE_criterion=infoNCE_criterion, chain_dataloader=chain_dataloader)
