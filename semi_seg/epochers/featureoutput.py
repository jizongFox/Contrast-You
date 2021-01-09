from itertools import repeat
from typing import List

import torch
from torch import Tensor, nn

from contrastyou.epocher._utils import preprocess_input_with_single_transformation  # noqa
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation  # noqa
from contrastyou.epocher._utils import write_predict, write_img_target  # noqa
from contrastyou.featextractor.unet import FeatureExtractor
from contrastyou.helper import average_iter, weighted_average_iter
from contrastyou.projectors.heads import ClusterHead  # noqa
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.meters2 import AverageValueMeter, MultipleAverageValueMeter, \
    MeterInterface
from deepclustering2.type import T_loss
from deepclustering2.utils import F
from semi_seg._utils import ClusterProjectorWrapper, IICLossWrapper, _filter_decodernames
from .miepocher import MITrainEpocher, ConsistencyMIEpocher


class _FeatureOutputIICEpocher:
    meters: MeterInterface
    _output_extractor: FeatureExtractor
    _fextractor: FeatureExtractor
    _feature_position: List[str]
    _feature_importance: List[float]
    _model: nn.Module

    def init(self, *, projectors_wrapper_output: ClusterProjectorWrapper, IIDSegCriterionWrapper_output: IICLossWrapper,
             **kwargs) -> None:
        super(_FeatureOutputIICEpocher, self).init(**kwargs)  # noqa
        self._projectors_wrapper_output = projectors_wrapper_output  # noqa
        self._IIDSegCriterionWrapper_output = IIDSegCriterionWrapper_output  # noqa

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)  # noqa
        meters.register_meter("foutmi", AverageValueMeter())
        meters.register_meter("foutindividual_mis", MultipleAverageValueMeter())
        return meters

    # extend the feature extractor
    def _run(self, *args, **kwargs):
        with  FeatureExtractor(self._model, "DeConv_1x1") as self._output_extractor:  # noqa
            return super(_FeatureOutputIICEpocher, self)._run(*args, **kwargs)  # noqa

    # we extend a method in order not to kill all.
    def regularization_on_feature_output(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int,
                                         *args, **kwargs):
        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2
        iic_losses_for_features = []
        unlabeled_preds = self._output_extractor["DeConv_1x1"][- n_uls:].softmax(1).detach()
        output_size = unlabeled_preds.shape[-2:]
        for i, (inter_feature, projector, criterion) \
            in enumerate(zip(self._fextractor, self._projectors_wrapper_output, self._IIDSegCriterionWrapper_output)):
            if isinstance(projector, ClusterHead):  # features from encoder
                continue
            # project unlabeled features to probabilities:
            unlabeled_features = inter_feature[- n_uls:]
            prob_list1 = [F.interpolate(x, size=output_size, mode="bilinear", align_corners=True) for x in
                          projector(unlabeled_features)]

            _iic_loss_list = [criterion(x, y) for x, y in zip(prob_list1, repeat(unlabeled_preds))]
            _iic_loss = average_iter(_iic_loss_list)
            iic_losses_for_features.append(_iic_loss)
        reg_loss = weighted_average_iter(iic_losses_for_features, self._feature_importance[1:])
        self.meters["foutmi"].add(-reg_loss.item())
        self.meters["foutindividual_mis"].add(**dict(zip(
            _filter_decodernames(self._feature_position),
            [-x.item() for x in iic_losses_for_features]
        )))

        return reg_loss


class FeatureOutputCrossMIEpocher(_FeatureOutputIICEpocher, MITrainEpocher):

    def init(self, *, projectors_wrapper: ClusterProjectorWrapper, projectors_wrapper_output: ClusterProjectorWrapper,
             # noqa
             IIDSegCriterionWrapper: IICLossWrapper, IIDSegCriterionWrapper_output: IICLossWrapper,  # noqa
             cross_reg_weight: float, output_reg_weight: float,  # noqa
             enforce_matching=False, **kwargs):  # noqa
        super().init(
            reg_weight=1.0, projectors_wrapper=projectors_wrapper,
            IIDSegCriterionWrapper=IIDSegCriterionWrapper, enforce_matching=enforce_matching,
            projectors_wrapper_output=projectors_wrapper_output,
            IIDSegCriterionWrapper_output=IIDSegCriterionWrapper_output,
            **kwargs
        )
        self._cross_reg_weight = cross_reg_weight  # noqa
        self._output_reg_weight = output_reg_weight  # noqa

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, *args, **kwargs):
        cross_mi = torch.tensor(0, dtype=torch.float, device=unlabeled_logits_tf.device)
        if self._cross_reg_weight > 0:
            cross_mi = super(FeatureOutputCrossMIEpocher, self).regularization(unlabeled_tf_logits,
                                                                               unlabeled_logits_tf,
                                                                               seed, *args, **kwargs)
        featureoutput_mi = torch.tensor(0, dtype=torch.float, device=unlabeled_logits_tf.device)
        if self._output_reg_weight > 0:
            featureoutput_mi = self.regularization_on_feature_output(unlabeled_tf_logits, unlabeled_logits_tf,
                                                                     seed, *args, **kwargs)
        return cross_mi * self._cross_reg_weight + featureoutput_mi * self._output_reg_weight


class FeatureOutputCrossIICUDAEpocher(_FeatureOutputIICEpocher, ConsistencyMIEpocher):

    def init(self, *, mi_weight: float, consistency_weight: float, output_reg_weight: float,  # noqa
             projectors_wrapper: ClusterProjectorWrapper, IIDSegCriterionWrapper: IICLossWrapper,
             reg_criterion: T_loss, enforce_matching=False, **kwargs):  # noqa
        super().init(mi_weight=mi_weight, consistency_weight=consistency_weight, projectors_wrapper=projectors_wrapper,
                     IIDSegCriterionWrapper=IIDSegCriterionWrapper, reg_criterion=reg_criterion,
                     enforce_matching=enforce_matching, **kwargs, )
        self._output_reg_weight = output_reg_weight  # noqa

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, *args, **kwargs):
        cross_udaiic = super().regularization(unlabeled_tf_logits, unlabeled_logits_tf, seed, *args, **kwargs)
        output_iic = torch.tensor(0.0, device=unlabeled_logits_tf.device, dtype=torch.float)
        if self._output_reg_weight > 0:
            output_iic = self.regularization_on_feature_output(unlabeled_tf_logits, unlabeled_logits_tf, seed, *args,
                                                               **kwargs)
        return cross_udaiic + self._output_reg_weight * output_iic
