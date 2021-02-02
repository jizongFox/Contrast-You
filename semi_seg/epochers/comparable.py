import math
from functools import lru_cache
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch import nn

from contrastyou.epocher._utils import unfold_position  # noqa
from contrastyou.featextractor.unet import FeatureExtractor
from contrastyou.helper import average_iter, weighted_average_iter
from contrastyou.projectors.heads import ProjectionHead, LocalProjectionHead
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats as disable_bn
from deepclustering2.loss import Entropy
from deepclustering2.meters2 import EpochResultDict, AverageValueMeter, MeterInterface, MultipleAverageValueMeter
from deepclustering2.models import ema_updater as EMA_Updater
from deepclustering2.schedulers.customized_scheduler import RampScheduler
from deepclustering2.type import T_loss
from semi_seg._utils import ContrastiveProjectorWrapper
from ._helper import unl_extractor, __AssertWithUnLabeledData
from .base import TrainEpocher
from .miepocher import MITrainEpocher, ConsistencyTrainEpocher


class MeanTeacherEpocher(TrainEpocher, __AssertWithUnLabeledData):

    def _init(self, *, reg_weight: float, teacher_model: nn.Module, reg_criterion: T_loss,  # noqa
              ema_updater: EMA_Updater, **kwargs):  # noqa
        super()._init(reg_weight=reg_weight, **kwargs)
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


class UCMeanTeacherEpocher(MeanTeacherEpocher, __AssertWithUnLabeledData):

    def _init(self, *, reg_weight: float, teacher_model: nn.Module, reg_criterion: T_loss, ema_updater: EMA_Updater,
              threshold: RampScheduler = None, **kwargs):
        super()._init(reg_weight=reg_weight, teacher_model=teacher_model, reg_criterion=reg_criterion,
                      ema_updater=ema_updater, **kwargs)
        assert isinstance(threshold, RampScheduler), threshold
        self._threshold: RampScheduler = threshold
        self._entropy_loss = Entropy(reduction="none")

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(UCMeanTeacherEpocher, self)._configure_meters(meters)
        meters.register_meter("uc_weight", AverageValueMeter())
        meters.register_meter("uc_ratio", AverageValueMeter())
        return meters

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int,
                       unlabeled_image: Tensor, unlabeled_image_tf: Tensor, *args, **kwargs):
        @torch.no_grad()
        def get_teacher_pred_with_tf(uimage, noise=None):
            if noise is not None:
                uimage += noise
            teacher_unlabeled_logit = self._teacher_model(uimage)
            with FixRandomSeed(seed):
                teacher_unlabeled_logit_tf = torch.stack(
                    [self._affine_transformer(x) for x in teacher_unlabeled_logit], dim=0)
            return teacher_unlabeled_logit_tf

        # compare teacher_unlabeled_logit_tf and student unlabeled_tf_logits
        self._reg_criterion.reduction = "none"  # here the self._reg_criterion should be nn.MSELoss or KLDiv()
        teacher_unlabeled_logit_tf = get_teacher_pred_with_tf(unlabeled_image)
        reg_loss = self._reg_criterion(unlabeled_tf_logits.softmax(1), teacher_unlabeled_logit_tf.softmax(1).detach())

        # uncertainty:
        with disable_bn(self._teacher_model):
            uncertainty_predictions = [
                get_teacher_pred_with_tf(unlabeled_image, 0.05 * torch.randn_like(unlabeled_image)) for _ in range(8)
            ]

        average_prediction = average_iter(uncertainty_predictions)

        entropy = self._entropy_loss(average_prediction.softmax(1)) / math.log(average_prediction.shape[1])
        th = self._threshold.value
        mask = (entropy <= th).float()

        self.meters["uc_weight"].add(th)
        self.meters["uc_ratio"].add(mask.mean().item())

        # update teacher model here.
        self._ema_updater(self._teacher_model, self._model)
        return (reg_loss.mean(1) * mask).mean()


class MIMeanTeacherEpocher(MITrainEpocher, __AssertWithUnLabeledData):

    def _init(self, *, mi_estimator_array: Iterable[Callable[[Tensor, Tensor], Tensor]],
              teacher_model: nn.Module = None,
              ema_updater: EMA_Updater = None, mt_weight: float = None, mi_weight: float = None, enforce_matching=False,
              reg_criterion: T_loss = None, **kwargs):
        super(MIMeanTeacherEpocher, self)._init(reg_weight=1.0, mi_estimator_array=mi_estimator_array,
                                                enforce_matching=enforce_matching, **kwargs)
        assert reg_criterion is not None
        assert teacher_model is not None
        assert ema_updater is not None
        assert mt_weight is not None
        assert mi_weight is not None

        self._reg_criterion = reg_criterion  # noqa
        self._teacher_model = teacher_model  # noqa
        self._ema_updater = ema_updater  # noqa
        self._mt_weight = float(mt_weight)  # noqa
        self._mi_weight = float(mi_weight)  # noqa

    def _set_model_state(self, model) -> None:
        model.train()
        self._teacher_model.train()

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(MIMeanTeacherEpocher, self)._configure_meters(meters)
        meters.register_meter("consistency", AverageValueMeter())
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        with FeatureExtractor(self._teacher_model, self._feature_position) as self._teacher_fextractor:  # noqa
            return super(MIMeanTeacherEpocher, self)._run()

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

        # clear feature cache
        self._teacher_fextractor.clear()
        with torch.no_grad():
            teacher_logits = self._teacher_model(unlabeled_image)
        with FixRandomSeed(seed):
            teacher_logits_tf = torch.stack([self._affine_transformer(x) for x in teacher_logits], dim=0)
        assert teacher_logits_tf.shape == teacher_logits.shape, (teacher_logits_tf.shape, teacher_logits.shape)

        def generate_iic(student_f, teacher_f, mi_estimator: Callable[[Tensor, Tensor], Tensor]):
            _, student_tf_features = torch.chunk(student_f, 2, dim=0)
            with FixRandomSeed(seed):
                teacher_f_tf = torch.stack([self._affine_transformer(x) for x in teacher_f], dim=0)

            assert teacher_f.shape == teacher_f_tf.shape, (teacher_f.shape, teacher_f_tf.shape)
            loss = mi_estimator(student_f, teacher_f_tf)
            return loss

        loss_list = [
            generate_iic(s, t, mi) for s, t, mi in zip(
                unl_extractor(self._fextractor, n_uls=n_uls),
                self._teacher_fextractor, self._mi_estimator_array)
        ]

        reg_loss = weighted_average_iter(loss_list, self._feature_importance)
        self.meters["mi"].add(-reg_loss.item())
        self.meters["individual_mis"].add(**dict(zip(
            self._feature_position,
            [-x.item() for x in loss_list]
        )))
        uda_loss = ConsistencyTrainEpocher.regularization(
            self,  # noqa
            unlabeled_tf_logits,
            teacher_logits_tf.detach(),
            seed,
        )

        # update ema
        self._ema_updater(self._teacher_model, self._model)

        return self._mt_weight * uda_loss + self._mi_weight * reg_loss


class MIDLPaperEpocher(ConsistencyTrainEpocher, __AssertWithUnLabeledData):

    def init(self, *, mi_weight: float, consistency_weight: float, iic_segcriterion: T_loss,  # noqa
             reg_criterion: T_loss,  # noqa
             **kwargs):  # noqa
        super().init(reg_weight=1.0, reg_criterion=reg_criterion, **kwargs)
        self._iic_segcriterion = iic_segcriterion  # noqa
        self._mi_weight = mi_weight  # noqa
        self._consistency_weight = consistency_weight  # noqa

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
        iic_loss = self._iic_segcriterion(unlabeled_tf_logits.softmax(1), unlabeled_logits_tf.softmax(1).detach())
        self.meters["iic_mi"].add(iic_loss.item())
        return uda_loss * self._consistency_weight + iic_loss * self._mi_weight


class EntropyMinEpocher(TrainEpocher, __AssertWithUnLabeledData):

    def init(self, *, reg_weight: float, **kwargs):
        super().init(reg_weight=reg_weight, **kwargs)
        self._entropy_criterion = Entropy()  # noqa

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


class InfoNCEEpocher(TrainEpocher, __AssertWithUnLabeledData):

    def _init(self, *, reg_weight: float, projectors_wrapper: ContrastiveProjectorWrapper = None,
              infoNCE_criterion: T_loss = None, **kwargs):
        assert projectors_wrapper is not None and infoNCE_criterion is not None, (projectors_wrapper, infoNCE_criterion)
        super()._init(reg_weight=reg_weight, **kwargs)
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

            proj_tf_feature, proj_feature_tf = self.unlabeled_projection(unl_features=features, projector=projector,
                                                                         seed=seed)

            if isinstance(projector, ProjectionHead):
                norm_tf_feature, norm_feature_tf = proj_tf_feature, proj_feature_tf
                assert len(norm_tf_feature.shape) == 2, norm_tf_feature.shape
                labels = self.global_label_generator(partition_list=partition_group, patient_list=label_group)

            elif isinstance(projector, LocalProjectionHead):
                proj_feature_tf_unfold, positional_label = unfold_position(proj_feature_tf,
                                                                           partition_num=proj_feature_tf.shape[-2:])
                proj_tf_feature_unfold, _ = unfold_position(proj_tf_feature, partition_num=proj_tf_feature.shape[-2:])

                __b = proj_tf_feature_unfold.size(0)
                norm_feature_tf = proj_feature_tf_unfold.view(__b, -1)
                norm_tf_feature = proj_tf_feature_unfold.view(__b, -1)

                labels = self.local_label_generator(partition_list=partition_group, patient_list=label_group,
                                                    location_list=positional_label)

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

    @property  # noqa
    @lru_cache()
    def global_label_generator(self):
        from contrastyou.epocher._utils import GlobalLabelGenerator  # noqa
        return GlobalLabelGenerator()

    @property  # noqa
    @lru_cache()
    def local_label_generator(self):
        from contrastyou.epocher._utils import LocalLabelGenerator  # noqa
        return LocalLabelGenerator()

    def unlabeled_projection(self, unl_features, projector, seed):
        unlabeled_features, unlabeled_tf_features = torch.chunk(unl_features, 2, dim=0)
        with FixRandomSeed(seed):
            unlabeled_features_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_features], dim=0)
        assert unlabeled_tf_features.shape == unlabeled_tf_features.shape, \
            (unlabeled_tf_features.shape, unlabeled_tf_features.shape)

        proj_tf_feature, proj_feature_tf = torch.chunk(
            projector(torch.cat([unlabeled_tf_features, unlabeled_features_tf], dim=0)), 2, dim=0
        )
        return proj_tf_feature, proj_feature_tf
    
    def _assertion(self):
        super(InfoNCEEpocher, self)._assertion()
