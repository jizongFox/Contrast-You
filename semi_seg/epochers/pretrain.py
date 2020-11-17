from contrastyou.arch.unet import freeze_grad
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation  # noqa
from contrastyou.epocher._utils import write_predict, write_img_target  # noqa
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.meters2 import EpochResultDict
from deepclustering2.type import T_loss
from .base import PretrainEpocher
from .comparable import InfoNCEEpocher
from .._utils import ContrastiveProjectorWrapper


class InfoNCEPretrainEpocher(PretrainEpocher, InfoNCEEpocher):

    def init(self, *, chain_dataloader=None, projectors_wrapper: ContrastiveProjectorWrapper = None,
             infoNCE_criterion: T_loss = None, **kwargs):
        super(InfoNCEPretrainEpocher, self).init(projectors_wrapper=projectors_wrapper,
                                                 infoNCE_criterion=infoNCE_criterion, chain_dataloader=chain_dataloader)

    def _run(self, *args, **kwargs) -> EpochResultDict:
        with freeze_grad(self._model, self._feature_position) as self._model:  # noqa
            return super(InfoNCEPretrainEpocher, self)._run(*args, **kwargs)
