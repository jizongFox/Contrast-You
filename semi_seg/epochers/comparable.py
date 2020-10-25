from functools import lru_cache

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from contrastyou.epocher._utils import preprocess_input_with_single_transformation  # noqa
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation  # noqa
from contrastyou.epocher._utils import write_predict, write_img_target, unfold_position  # noqa
from contrastyou.helper import average_iter, weighted_average_iter
from contrastyou.projectors.heads import ProjectionHead, LocalProjectionHead
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.loss import Entropy
from deepclustering2.meters2 import EpochResultDict, AverageValueMeter, MeterInterface, MultipleAverageValueMeter
from deepclustering2.models import ema_updater as EMA_Updater
from deepclustering2.type import T_loss
from semi_seg._utils import ClusterProjectorWrapper, PICALossWrapper, FeatureExtractor, IICLossWrapper, \
    ContrastiveProjectorWrapper
from .base import TrainEpocher
from .helper import unl_extractor
from .miepocher import IICTrainEpocher, UDATrainEpocher


class MeanTeacherEpocher(TrainEpocher):

    def init(self, *, reg_weight: float, teacher_model: nn.Module, reg_criterion: T_loss,  # noqa
             ema_updater: EMA_Updater, **kwargs):  # noqa
        super().init(reg_weight=reg_weight, **kwargs)
        self._reg_criterion = reg_criterion  # noqa
        self._teacher_model = teacher_model  # noqa
        self._ema_updater = ema_updater  # noqa
        self._model.train()
        self._teacher_model.train()

    def regularization(
        self,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed: int,
        unlabeled_image: Tensor,
        unlabeled_image_tf: Tensor, *args, **kwargs
    ):
        with torch.no_grad():
            teacher_unlabeled_logit = self._teacher_model(unlabeled_image)
        with FixRandomSeed(seed):
            teacher_unlabeled_logit_tf = torch.stack(
                [self._affine_transformer(x) for x in teacher_unlabeled_logit], dim=0)

        # compare teacher_unlabeled_logit_tf and student unlabeled_tf_logits
        reg_loss = self._reg_criterion(unlabeled_tf_logits.softmax(1), teacher_unlabeled_logit_tf.softmax(1).detach())
        # update teacher model here.
        self._ema_updater(self._teacher_model, self._model)
        return reg_loss


class IICMeanTeacherEpocher(IICTrainEpocher):

    def init(self, *, projectors_wrapper: ClusterProjectorWrapper, IIDSegCriterionWrapper: IICLossWrapper,
             enforce_matching=False, reg_criterion: T_loss = None, teacher_model: nn.Module = None,
             ema_updater: EMA_Updater = None, mt_weight: float = None, iic_weight: float = None, **kwargs):
        super().init(reg_weight=1.0, projectors_wrapper=projectors_wrapper,
                     IIDSegCriterionWrapper=IIDSegCriterionWrapper, enforce_matching=enforce_matching, **kwargs)
        assert self._reg_weight == 1.0, self._reg_weight
        assert reg_criterion is not None
        assert teacher_model is not None
        assert ema_updater is not None
        assert mt_weight is not None
        assert iic_weight is not None

        self._reg_criterion = reg_criterion  # noqa
        self._teacher_model = teacher_model  # noqa
        self._ema_updater = ema_updater  # noqa
        self._mt_weight = float(mt_weight)  # noqa
        self._iic_weight = float(iic_weight)  # noqa
        self._teacher_model.train()
        self._model.train()

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(IICMeanTeacherEpocher, self)._configure_meters(meters)
        meters.register_meter("uda", AverageValueMeter())
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        with FeatureExtractor(self._teacher_model, self._feature_position) as self._teacher_fextractor:  # noqa
            return super(IICMeanTeacherEpocher, self)._run()

    def regularization(
        self,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed: int,
        unlabeled_image: Tensor = None,
        unlabeled_image_tf: Tensor = None,
        *args, **kwargs
    ):
        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2

        with torch.no_grad():
            teacher_logits = self._teacher_model(unlabeled_image)
        with FixRandomSeed(seed):
            teacher_logits_tf = torch.stack([self._affine_transformer(x) for x in teacher_logits], dim=0)
        assert teacher_logits_tf.shape == teacher_logits.shape, (teacher_logits_tf.shape, teacher_logits.shape)

        def generate_iic(student_f, teacher_f, projector, criterion):
            _, student_tf_features = torch.chunk(student_f, 2, dim=0)
            with FixRandomSeed(seed):
                teacher_f_tf = torch.stack([self._affine_transformer(x) for x in teacher_f], dim=0)

            assert teacher_f.shape == teacher_f_tf.shape, (teacher_f.shape, teacher_f_tf.shape)
            prob1, prob2 = list(
                zip(*[torch.chunk(x, 2, 0) for x in projector(
                    torch.cat([teacher_f_tf, student_tf_features], dim=0)
                )])
            )
            loss = average_iter([criterion(x, y) for x, y in zip(prob1, prob2)])
            return loss

        loss_list = [
            generate_iic(s, t, p, c) for s, t, p, c in zip(
                unl_extractor(self._fextractor, n_uls=n_uls),
                self._teacher_fextractor, self._projectors_wrapper,
                self._IIDSegCriterionWrapper)
        ]

        reg_loss = weighted_average_iter(loss_list, self._feature_importance)
        self.meters["mi"].add(-reg_loss.item())
        self.meters["individual_mis"].add(**dict(zip(
            self._feature_position,
            [-x.item() for x in loss_list]
        )))
        uda_loss = UDATrainEpocher.regularization(
            self,  # noqa
            unlabeled_tf_logits,
            teacher_logits_tf.detach(),
            seed,
        )

        # update ema
        self._ema_updater(self._teacher_model, self._model)

        return self._mt_weight * uda_loss + self._iic_weight * reg_loss


class MIDLPaperEpocher(UDATrainEpocher):

    def init(self, *, iic_weight: float, uda_weight: float, iic_segcriterion: T_loss, reg_criterion: T_loss,  # noqa
             **kwargs):  # noqa
        super().init(reg_weight=1.0, reg_criterion=reg_criterion, **kwargs)
        self._iic_segcriterion = iic_segcriterion
        self._iic_weight = iic_weight
        self._uda_weight = uda_weight

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(MIDLPaperEpocher, self)._configure_meters(meters)
        meters.register_meter("iic_mi", AverageValueMeter())
        return meters

    def regularization(
        self,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed, *args, **kwargs
    ):
        uda_loss = super(MIDLPaperEpocher, self).regularization(
            unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_logits_tf=unlabeled_logits_tf,
            seed=seed, *args, **kwargs
        )
        iic_loss = self._iic_segcriterion(unlabeled_tf_logits, unlabeled_logits_tf.detach())
        self.meters["iic_mi"].add(iic_loss.item())
        return uda_loss * self._uda_weight + iic_loss * self._iic_weight


class EntropyMinEpocher(TrainEpocher):

    def init(self, *, reg_weight: float, **kwargs):
        super().init(reg_weight=reg_weight, **kwargs)
        self._entropy_criterion = Entropy()

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(EntropyMinEpocher, self)._configure_meters(meters)
        meters.register_meter("entropy", AverageValueMeter())
        return meters

    def regularization(
        self,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        seed, *args, **kwargs
    ):
        reg_loss = self._entropy_criterion(unlabeled_logits_tf.softmax(1))
        self.meters["entropy"].add(reg_loss.item())
        return reg_loss


class PICAEpocher(IICTrainEpocher):

    def init(self, *, reg_weight: float, projectors_wrapper: ClusterProjectorWrapper,
             PICASegCriterionWrapper: PICALossWrapper, enforce_matching=False, **kwargs):
        super().init(reg_weight=reg_weight, projectors_wrapper=projectors_wrapper,
                     IIDSegCriterionWrapper=PICASegCriterionWrapper,  # noqa
                     enforce_matching=enforce_matching, **kwargs)


class InfoNCEEpocher(TrainEpocher):

    def init(self, *, reg_weight: float, projectors_wrapper: ContrastiveProjectorWrapper = None,
             infoNCE_criterion: T_loss = None, **kwargs):
        assert projectors_wrapper is not None and infoNCE_criterion is not None
        super().init(reg_weight=reg_weight, **kwargs)
        self._projectors_wrapper: ContrastiveProjectorWrapper = projectors_wrapper  # noqa
        self._infonce_criterion: T_loss = infoNCE_criterion  # noqa

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("mi", AverageValueMeter())
        meters.register_meter("individual_mis", MultipleAverageValueMeter())
        return meters

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, label_group,
                       partition_group, *args, **kwargs):
        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2

        def generate_infonce(features, projector):
            unlabeled_features, unlabeled_tf_features = torch.chunk(features, 2, dim=0)
            with FixRandomSeed(seed):
                unlabeled_features_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_features], dim=0)
            assert unlabeled_tf_features.shape == unlabeled_tf_features.shape, \
                (unlabeled_tf_features.shape, unlabeled_tf_features.shape)
            proj_tf_feature, proj_feature_tf = torch.chunk(
                projector(torch.cat([unlabeled_tf_features, unlabeled_features_tf], dim=0)), 2, dim=0
            )
            # normalization and label generation goes differently here.
            if isinstance(projector, ProjectionHead):
                norm_tf_feature, norm_feature_tf = F.normalize(proj_tf_feature, p=2, dim=1), \
                                                   F.normalize(proj_feature_tf, p=2, dim=1)
                assert len(norm_tf_feature.shape) == 2, norm_tf_feature.shape
                labels = self.global_label_generator(partition_list=partition_group, patient_list=label_group)
            elif isinstance(projector, LocalProjectionHead):
                proj_feature_tf_unfold, positional_label = unfold_position(proj_feature_tf,
                                                                           partition_num=proj_feature_tf.shape[-2:])
                proj_tf_feature_unfold, _ = unfold_position(proj_tf_feature, partition_num=proj_tf_feature.shape[-2:])

                __b = proj_tf_feature_unfold.size(0)

                labels = self.local_label_generator(partition_list=partition_group, patient_list=label_group,
                                                    location_list=positional_label)
                norm_feature_tf = F.normalize(proj_feature_tf_unfold.view(__b, -1), dim=1, p=2)
                norm_tf_feature = F.normalize(proj_tf_feature_unfold.view(__b, -1), dim=1, p=2)

            else:
                raise NotImplementedError(type(projector))
            return self._infonce_criterion(torch.stack([norm_feature_tf, norm_tf_feature], dim=1), labels=labels)

        losses = [generate_infonce(f, p) for f, p in zip(unl_extractor(self._fextractor, n_uls=n_uls),
                                                         self._projectors_wrapper)]

        reg_loss = weighted_average_iter(losses, self._feature_importance)
        self.meters["mi"].add(-reg_loss.item())
        self.meters["individual_mis"].add(**dict(zip(
            self._feature_position,
            [-x.item() for x in losses]
        )))
        return reg_loss

    @property
    @lru_cache()
    def global_label_generator(self):
        from contrastyou.epocher._utils import GlobalLabelGenerator  # noqa
        return GlobalLabelGenerator()

    @property
    @lru_cache()
    def local_label_generator(self):
        from contrastyou.epocher._utils import LocalLabelGenerator  # noqa
        return LocalLabelGenerator()
