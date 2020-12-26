from contrastyou.epocher._utils import preprocess_input_with_single_transformation  # noqa
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation  # noqa
from contrastyou.epocher._utils import write_predict, write_img_target  # noqa
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats as disable_bn
from deepclustering2.epoch import _Epocher  # noqa
from semi_seg.epochers.base import EvalEpocher


class EvalEpocherWOEval(EvalEpocher):
    """
    This epocher is set to using the current estimation of batch and without accumulating the statistic
    network in train mode while BN is in disable accumulation mode.
    Usually improves performance with some domain gap
    """

    def _run(self, *args, **kwargs):
        with disable_bn(self._model):  # disable bn accumulation
            return super(EvalEpocherWOEval, self)._run(*args, **kwargs)

    def _set_model_state(self, model) -> None:
        model.train()
