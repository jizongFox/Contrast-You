import math
from abc import abstractmethod
from functools import lru_cache, partial
from typing import Callable, Iterable

import torch
from loguru import logger
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from contrastyou.epocher._utils import unfold_position  # noqa
from contrastyou.featextractor.unet import FeatureExtractor
from contrastyou.helper import average_iter, weighted_average_iter
from contrastyou.losses.contrast_loss import is_normalized
from contrastyou.losses.iic_loss import _ntuple
from contrastyou.projectors.heads import ProjectionHead
from contrastyou.projectors.nn import Normalize
from deepclustering2.configparser._utils import get_config  # noqa
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats as disable_bn  # noqa
from deepclustering2.loss import Entropy
from deepclustering2.meters2 import EpochResultDict, AverageValueMeter, MeterInterface, MultipleAverageValueMeter
from deepclustering2.models import ema_updater as EMA_Updater
from deepclustering2.schedulers.customized_scheduler import RampScheduler
from deepclustering2.type import T_loss
from semi_seg._utils import ContrastiveProjectorWrapper
from ._helper import unl_extractor, __AssertWithUnLabeledData, _FeatureExtractorMixin
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


class _InfoNCEBasedEpocher(_FeatureExtractorMixin, TrainEpocher, __AssertWithUnLabeledData):
    """base epocher class for infonce like method"""

    def _init(self, *, reg_weight: float, projectors_wrapper: ContrastiveProjectorWrapper = None,
              infoNCE_criterion: T_loss = None, **kwargs):
        assert projectors_wrapper is not None and infoNCE_criterion is not None, (projectors_wrapper, infoNCE_criterion)
        super()._init(reg_weight=reg_weight, **kwargs)
        assert projectors_wrapper is not None and infoNCE_criterion is not None, (projectors_wrapper, infoNCE_criterion)

        self._projectors_wrapper: ContrastiveProjectorWrapper = projectors_wrapper  # noqa
        self._infonce_criterion: T_loss = infoNCE_criterion  # noqa
        self.__encoder_method_name__ = "supcontrast"

    def set_global_contrast_method(self, *, method_name):
        assert method_name in ("simclr", "supcontrast")
        self.__encoder_method_name__ = method_name
        logger.debug("{} set global contrast method to be {}", self.__class__.__name__, method_name)

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("mi", AverageValueMeter())
        meters.register_meter("individual_mis", MultipleAverageValueMeter())
        return meters

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

    @lru_cache()
    def global_label_generator(self, gmethod: str):
        from contrastyou.epocher._utils import GlobalLabelGenerator  # noqa
        logger.debug("initialize {} label generator for encoder training", self.__encoder_method_name__)
        if gmethod == "supcontrast":
            return GlobalLabelGenerator()
        elif gmethod == "simclr":
            return GlobalLabelGenerator(True, True)
        else:
            raise NotImplementedError(gmethod)

    @lru_cache()
    def local_label_generator(self):
        from contrastyou.epocher._utils import LocalLabelGenerator  # noqa
        return LocalLabelGenerator()

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, label_group,
                       partition_group, *args, **kwargs):
        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2

        losses = [
            self.generate_infonce(
                feature_name=n, features=f, projector=p, seed=seed, partition_group=partition_group,
                label_group=label_group) for n, f, p in
            zip(self._fextractor.feature_names, unl_extractor(self._fextractor, n_uls=n_uls), self._projectors_wrapper)
        ]
        reg_loss = weighted_average_iter(losses, self._feature_importance)
        self.meters["mi"].add(-reg_loss.item())
        self.meters["individual_mis"].add(**dict(zip(
            [f"{p}|{i}" for i, p in enumerate(self._feature_position)],
            [-x.item() for x in losses]
        )))
        return reg_loss

    def generate_infonce(self, *, feature_name, features, projector, seed, partition_group, label_group) -> Tensor:
        proj_tf_feature, proj_feature_tf = self.unlabeled_projection(unl_features=features, projector=projector,
                                                                     seed=seed)

        if isinstance(projector, ProjectionHead):
            # it goes to **global** representation here.
            return self._global_infonce(
                feature_name=feature_name,
                proj_tf_feature=proj_tf_feature,
                proj_feature_tf=proj_feature_tf,
                partition_group=partition_group,
                label_group=label_group
            )
        # it goes to a **dense** representation on pixels
        return self._dense_based_infonce(
            feature_name=feature_name,
            proj_tf_feature=proj_tf_feature,
            proj_feature_tf=proj_feature_tf,
            partition_group=partition_group,
            label_group=label_group
        )

    @abstractmethod
    def _global_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                        label_group) -> Tensor:
        ...

    @abstractmethod
    def _dense_based_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                             label_group) -> Tensor:
        ...

    @staticmethod
    def _reshape_dense_feature(proj_tf_feature: Tensor, proj_feature_tf: Tensor):
        """reshape a feature map from [b,c h,w] to [b, hw, c]"""
        b, c, *hw = proj_tf_feature.shape
        proj_tf_feature = proj_tf_feature.view(b, c, -1).permute(0, 2, 1)
        proj_feature_tf = proj_feature_tf.view(b, c, -1).permute(0, 2, 1)
        return proj_tf_feature, proj_feature_tf


class InfoNCEEpocher(_InfoNCEBasedEpocher):
    """INFONCE that implements SIMCLR and SupContrast"""
    from contrastyou.losses.contrast_loss import SupConLoss2
    _infonce_criterion: SupConLoss2

    def _assertion(self):
        from contrastyou.losses.contrast_loss import SupConLoss2
        if not isinstance(self._infonce_criterion, SupConLoss2):
            raise RuntimeError(f"{self.__class__.__name__} only support `SupConLoss2`, "
                               f"given {type(self._infonce_criterion)}")
        super(InfoNCEEpocher, self)._assertion()

    def _global_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                        label_group) -> Tensor:
        """methods go for global vectors"""
        assert len(proj_tf_feature.shape) == 2, proj_tf_feature.shape
        assert is_normalized(proj_feature_tf) and is_normalized(proj_feature_tf)

        # generate simclr or supcontrast labels
        labels = self.global_label_generator(self.__encoder_method_name__)(partition_list=partition_group,
                                                                           patient_list=label_group)
        return self._infonce_criterion(proj_feature_tf, proj_tf_feature, target=labels)

    def _dense_based_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                             label_group) -> Tensor:
        if "Conv" in feature_name:
            # this is the dense feature from encoder
            return self._dense_infonce_for_encoder(
                feature_name=feature_name,
                proj_tf_feature=proj_tf_feature,
                proj_feature_tf=proj_feature_tf,
                partition_group=partition_group,
                label_group=label_group
            )
        return self._dense_infonce_for_decoder(
            feature_name=feature_name,
            proj_tf_feature=proj_tf_feature,
            proj_feature_tf=proj_feature_tf,
            partition_group=partition_group,
            label_group=label_group
        )

    def _dense_infonce_for_encoder(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                                   label_group):
        """here the dense prediction does not consider the spatial neighborhood"""
        # the mask of this dense metric would be image-wise simclr
        # usually the spatial size of feature map is very small
        assert "Conv" in feature_name, feature_name

        proj_tf_feature, proj_feature_tf = self._reshape_dense_feature(proj_tf_feature, proj_feature_tf)

        b, hw, c = proj_feature_tf.shape

        if not (is_normalized(proj_feature_tf, dim=2) and is_normalized(proj_feature_tf, dim=2)):
            proj_feature_tf = Normalize(dim=2)(proj_feature_tf)
            proj_tf_feature = Normalize(dim=2)(proj_tf_feature)

        _config = get_config(scope="base")
        include_all = _config["InfoNCEParameters"]["DenseParams"].get("include_all", False)
        if include_all:
            return self._infonce_criterion(proj_feature_tf.reshape(-1, c), proj_tf_feature.reshape(-1, c))
        else:
            raise RuntimeError("experimental results show that using batch-wise is better")

    def _dense_infonce_for_decoder(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                                   label_group):
        """here the dense predictions consider the neighborhood information, and the content similarity"""
        assert "Up" in feature_name, feature_name
        b, c, *hw = proj_feature_tf.shape
        output_size = (9, 9)
        sampled_norm_tf_feature, sampled_norm_feature_tf = self._feature_map_tailoring(
            proj_tf_feature=proj_tf_feature,
            proj_feature_tf=proj_feature_tf,
            output_size=output_size
        )
        assert sampled_norm_tf_feature.shape == torch.Size([b, c, *output_size])
        assert is_normalized(sampled_norm_tf_feature) and is_normalized(sampled_norm_feature_tf)

        # in order to save memory, we try to sample 1 positive pair with several negative pairs
        mask = self.generate_relation_masks(output_size)
        n_tf_feature, n_feature_tf = self._reshape_dense_feature(sampled_norm_tf_feature, sampled_norm_feature_tf)
        return sum([self._infonce_criterion(f1, f2, mask=mask) for f1, f2 in zip(n_tf_feature, n_feature_tf)]) / b

    def _feature_map_tailoring(self, *, proj_tf_feature: Tensor, proj_feature_tf: Tensor, output_size=(9, 9),
                               method="adaptive_avg"):
        """
        it consists of
        1. downsampling the feature map to a pre-defined size
        2. sampling fixed position
        3. reshaping the feature map
        4. create a relation mask
        """
        b, c, *hw = proj_feature_tf.shape
        # 1. upsampling
        proj_feature_tf = self._resize_featuremap(output_size=output_size, method=method)(proj_feature_tf)
        proj_tf_feature = self._resize_featuremap(output_size=output_size, method=method)(proj_tf_feature)
        # output features are [b,c,h_,w_] with h_, w_ as the reduced size
        if not (is_normalized(proj_feature_tf, dim=1) and is_normalized(proj_feature_tf, dim=1)):
            proj_feature_tf = Normalize(dim=1)(proj_feature_tf)
            proj_tf_feature = Normalize(dim=1)(proj_tf_feature)
        return proj_tf_feature, proj_feature_tf

    @lru_cache()
    def _resize_featuremap(self, output_size, method="adaptive_avg"):
        if method == "bilinear":
            return partial(F.interpolate, size=output_size, align_corners=True)
        elif method == "adaptive_avg":
            return nn.AdaptiveAvgPool2d(output_size=output_size)
        elif method == "adaptive_max":
            return nn.AdaptiveMaxPool2d(output_size=output_size)
        else:
            raise ValueError(method)

    @lru_cache()
    def generate_relation_masks(self, output_size) -> Tensor:
        _pair = _ntuple(2)
        output_size = _pair(output_size)
        size = output_size[0] * output_size[1]
        mask = torch.zeros(size, size, dtype=torch.float, device=self._device)
        for i in range(output_size[0]):
            for j in range(output_size[1]):
                relation = torch.ones(*output_size, dtype=torch.float, device=self._device) * -1
                relation[
                max(i - 1, 0):i + 2,
                max(j - 1, 0):j + 2
                ] = 0
                relation[i, j] = 1
                mask[i * output_size[0] + j] = relation.view(1, -1)
        return mask
