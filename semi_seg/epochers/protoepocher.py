from semi_seg._utils import ClusterProjectorWrapper, PICALossWrapper

from .miepocher import IICTrainEpocher


class PICAEpocher(IICTrainEpocher):

    def init(self, *, reg_weight: float, projectors_wrapper: ClusterProjectorWrapper, PICASegCriterionWrapper: PICALossWrapper,
             enforce_matching=False, **kwargs):
        super().init(reg_weight=reg_weight, projectors_wrapper=projectors_wrapper,
                     IIDSegCriterionWrapper=PICASegCriterionWrapper,
                     enforce_matching=enforce_matching, **kwargs)


