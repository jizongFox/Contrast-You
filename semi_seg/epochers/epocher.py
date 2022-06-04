import random
import typing as t
from abc import ABC
from contextlib import nullcontext
from copy import deepcopy
from functools import lru_cache, partial
from typing import Any, Dict, Optional, final

import rising.random as rr
import rising.transforms as rt
import torch
from loguru import logger
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from contrastyou.epochers.base import EpocherBase as _EpocherBase
from contrastyou.losses import LossClass
from contrastyou.meters import MeterInterface, UniversalDice, AverageValueMeter, SurfaceMeter
from contrastyou.types import criterionType, optimizerType, dataIterType, SizedIterable, CriterionType
from contrastyou.utils import get_dataset, class_name, fix_all_seed_for_transforms, get_lrs_from_optimizer
from contrastyou.utils.general import class2one_hot
from contrastyou.utils.utils import disable_tracking_bn_stats, get_model, ignore_exception
from semi_seg.augment import RisingWrapper
from semi_seg.epochers.helper import preprocess_input_with_twice_transformation, \
    preprocess_input_with_single_transformation, InferenceSaver
from semi_seg.hooks import EMAUpdater

if t.TYPE_CHECKING:
    from contrastyou.trainer.base import Trainer


def assert_transform_freedom(dataloader, is_true):
    dataset = get_dataset(dataloader)
    transform = dataset.transforms
    assert transform._total_freedom is is_true  # noqa


class EpocherBase(_EpocherBase, ABC):
    """ EpocherBase class to control the behavior of the training within one epoch.
    we add some control flow for sanity check
    >>> hooks = ...
    >>> epocher = EpocherBase(...)
    >>> epocher.init() # configuration check
    >>> with epocher.register_hook(*hooks):
    >>>     epocher.run(...)
    >>> epocher_result, best_score = epocher.get_metric(), epocher.get_score()
    """

    @property
    def num_classes(self):
        return get_model(self._model).num_classes

    def __init__(self, *, model: nn.Module, num_batches: int, cur_epoch=0, device="cpu", scaler: GradScaler,
                 **kwargs) -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device, scaler=scaler,
                         **kwargs)
        self._retain_graph = False

    def init(self, trainer: "Trainer" = None) -> None:
        """we added an assertion to control the hyper-parameters."""
        super(EpocherBase, self).init(trainer=trainer)
        self._assertion()

    @property
    def retain_graph(self):
        return self._retain_graph

    @retain_graph.setter
    def retain_graph(self, enable):
        logger.trace(f"retain_graph = {enable}")
        self._retain_graph = enable

    @final
    def run(self):
        if not self._initialized:
            raise RuntimeError(f"Call {class_name(self)}.init() before {class_name(self)}.run()")
        return super(EpocherBase, self).run()

    def _batch_update(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        returning the predicted dictionary, to pass to possible metric meter.
        """
        ...

    def batch_update(self, **kwargs) -> Optional[Dict[str, Any]]:
        """batch updater given labeled and unlabeled images, including model update, zero grad and optimizer steps.
        Gan style update should be included in this function as well."""
        for h in self._hooks:
            h.call_before_batch_update(**kwargs)
        predicted_result_dict = self._batch_update(**kwargs)
        for h in self._hooks:
            h.call_after_batch_update(**kwargs, result_dict=predicted_result_dict)
        return predicted_result_dict

    def _assertion(self):
        pass

    def forward_pass(self, **kwargs):
        for h in self._hooks:
            h.call_before_forward_pass(**kwargs)
        result = self._forward_pass(**kwargs)
        for h in self._hooks:
            h.call_after_forward_pass(**kwargs, result_dict=result)
        return result

    def _forward_pass(self, **kwargs) -> Any:
        ...

    def regularization(self, **kwargs):
        for h in self._hooks:
            h.call_before_regularization(**kwargs)
        result = self._regularization(**kwargs)
        for h in self._hooks:
            h.call_after_regularization(**kwargs, result_dict=result)
        return result

    def _regularization(self, **kwargs):
        return torch.tensor(0, dtype=torch.float, device=self._device)


class EvalEpocher(EpocherBase):
    meter_focus = "eval"

    def get_score(self) -> float:
        metric = self.meters["dice"].summary()
        return metric["DSC_mean"]

    def __init__(self, *, model: nn.Module, loader: DataLoader, sup_criterion: LossClass, cur_epoch=0, device="cpu",
                 scaler: GradScaler, accumulate_iter: int) -> None:
        super().__init__(model=model, num_batches=len(loader), cur_epoch=cur_epoch, device=device, scaler=scaler,
                         accumulate_iter=accumulate_iter)
        self._loader = loader
        self._sup_criterion: LossClass = sup_criterion

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("loss", AverageValueMeter())
        meters.register_meter("dice", UniversalDice(C, report_axis=report_axis))
        return meters

    def _run(self, **kwargs):
        self._model.eval()
        return self._run_implement()

    @torch.no_grad()
    def _run_implement(self):
        for i, eval_data in zip(self.indicator, self._loader):
            eval_img, eval_target, file_path, _, group = self._unzip_data(eval_data, self._device)
            self.batch_update(eval_img=eval_img,
                              eval_target=eval_target,
                              eval_group=group,
                              file_names=file_path)

            report_dict = self.meters.statistics()
            self.indicator.set_postfix_statics2(report_dict, force_update=True)

    def _batch_update(self, *, eval_img, eval_target, eval_group, file_names):
        with self.autocast:
            eval_logits = self._model(eval_img)
            onehot_target = class2one_hot(eval_target.squeeze(1), self.num_classes)
            eval_loss = self._sup_criterion(eval_logits.softmax(1), onehot_target, disable_assert=True)

        self.meters["loss"].add(eval_loss.item())
        self.meters["dice"].add(eval_logits.max(1)[1], eval_target.squeeze(1), group_name=eval_group)

    @staticmethod
    def _unzip_data(data, device):
        image, target, filename, partition, group = preprocess_input_with_single_transformation(data, device)
        return image, target, filename, partition, group


class InferenceEpocher(EvalEpocher):
    meter_focus = "infer"

    def __init__(self, *, model: nn.Module, loader: DataLoader, sup_criterion: LossClass, cur_epoch=0, device="cpu",
                 scaler: GradScaler, accumulate_iter: int, enable_prediction_saver: bool = True) -> None:
        super().__init__(model=model, loader=loader, sup_criterion=sup_criterion, cur_epoch=cur_epoch, device=device,
                         scaler=scaler, accumulate_iter=accumulate_iter)
        self.enable_prediction_saver = enable_prediction_saver

    def _run(self, **kwargs):
        self.saver = InferenceSaver(enable=self.enable_prediction_saver, save_dir=self.trainer.absolute_save_dir)
        return super()._run(**kwargs)

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super().configure_meters(meters)
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("ASD", SurfaceMeter(C=C, report_axises=report_axis, metername="average_surface"))
        return meters

    def _batch_update(self, *, eval_img, eval_target, eval_group, file_names):
        with self.autocast:
            eval_logits = self._model(eval_img)
            onehot_target = class2one_hot(eval_target.squeeze(1), self.num_classes)
            eval_loss = self._sup_criterion(eval_logits.softmax(1), onehot_target, disable_assert=True)
            self.saver(image=eval_img, target=eval_target, predict_logit=eval_logits, filenames=file_names)

        self.meters["loss"].add(eval_loss.item())
        self.meters["dice"].add(eval_logits.max(1)[1], eval_target.squeeze(1), group_name=eval_group)
        with ignore_exception():
            self.meters["ASD"].add(eval_logits.max(1)[1][None, ...], eval_target.squeeze(1)[None, ...])


class SemiSupervisedEpocher(EpocherBase, ABC):
    meter_focus = "semi"

    def _assertion(self):
        assert_transform_freedom(self._labeled_loader, False)
        if self._unlabeled_loader is not None:
            assert_transform_freedom(self._unlabeled_loader, False)

    def __init__(self, *, model: nn.Module, optimizer: optimizerType, labeled_loader: SizedIterable,
                 unlabeled_loader: SizedIterable, sup_criterion: criterionType, num_batches: int, cur_epoch=0,
                 device="cpu", two_stage: bool = False, disable_bn: bool = False, scaler: GradScaler,
                 accumulate_iter: int = 1, **kwargs) -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device, scaler=scaler,
                         accumulate_iter=accumulate_iter)
        self._optimizer = optimizer
        self._labeled_loader: SizedIterable = labeled_loader
        self._unlabeled_loader: SizedIterable = unlabeled_loader
        self._sup_criterion = sup_criterion

        self._affine_transformer = RisingWrapper(
            geometry_transform=rt.Compose(rt.BaseAffine(
                scale=rr.UniformParameter(0.8, 1.3),
                rotation=rr.UniformParameter(-45, 45),
                translation=rr.UniformParameter(-0.1, 0.1),
                degree=True,
                interpolation_mode="nearest",
                grad=True,
            ),
                rt.Mirror(dims=rr.DiscreteParameter([0, 1]), p_sample=0.9, grad=True)
            ),
            intensity_transform=rt.GammaCorrection(gamma=rr.UniformParameter(0.5, 2), grad=True)
        )
        self._two_stage = two_stage
        logger.opt(depth=1).trace("{} set to be using {} stage training", self.__class__.__name__,
                                  "two" if self._two_stage else "single")
        self._disable_bn = disable_bn
        if self._disable_bn:
            logger.debug("{} set to disable bn tracking", self.__class__.__name__)

        self.cur_batch_num = 0

    def transform_with_seed(self, features, *, mode: str, seed: int):
        assert mode in {"image", "feature"}, f"mode must be either `image` or `feature`, given {mode}"

        with fix_all_seed_for_transforms(seed):
            features_tf = self._affine_transformer(features, mode=mode, seed=seed)
        return features_tf

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(SemiSupervisedEpocher, self).configure_meters(meters)
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("sup_dice", UniversalDice(C, report_axis=report_axis))
        meters.register_meter("reg_loss", AverageValueMeter())
        return meters

    def _run(self, **kwargs):
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        self._model.train()
        return self._run_implement()

    def _run_implement(self):
        if len(self._unlabeled_loader) == 0:
            # in a fully supervised setting
            # maybe not necessary to control the randomness?
            self._unlabeled_loader = self._labeled_loader
        for self.cur_batch_num, labeled_data, unlabeled_data in zip(self.indicator, self._labeled_loader,
                                                                    self._unlabeled_loader):
            seed = random.randint(0, int(1e7))
            (labeled_image, _), labeled_target, labeled_filename, _, label_group = \
                self._unzip_data(labeled_data, self._device)
            (unlabeled_image, unlabeled_image_cf), _, unlabeled_filename, unl_partition, unl_group = \
                self._unzip_data(unlabeled_data, self._device)

            unlabeled_image_tf = self.transform_with_seed(unlabeled_image_cf, seed=seed, mode="image")

            self.batch_update(cur_batch_num=self.cur_batch_num,
                              labeled_image=labeled_image,
                              labeled_target=labeled_target,
                              labeled_filename=labeled_filename,
                              label_group=label_group, unlabeled_image=unlabeled_image,
                              unlabeled_image_tf=unlabeled_image_tf,
                              seed=seed, unl_group=unl_group, unl_partition=unl_partition,
                              unlabeled_filename=unlabeled_filename,
                              retain_graph=self._retain_graph)

            report_dict = self.meters.statistics()
            self.indicator.set_postfix_statics2(report_dict, force_update=self.cur_batch_num == self.num_batches - 1)

    def _batch_update(self, *, cur_batch_num: int, labeled_image, labeled_target, labeled_filename, label_group,
                      unlabeled_image, unlabeled_image_tf, seed, unl_group, unl_partition, unlabeled_filename,
                      retain_graph=False,
                      **kwargs):
        self.optimizer_zero(self._optimizer, cur_iter=cur_batch_num)
        if cur_batch_num <= 5:
            logger.trace(
                f"labeled_image filenames: {','.join(labeled_filename)}, "
                f"unlabeled_image filenames: {','.join(unlabeled_filename)}")

        with self.autocast:
            label_logits, unlabeled_logits, unlabeled_tf_logits = self.forward_pass(
                labeled_image=labeled_image,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf
            )

            unlabeled_logits_tf = self.transform_with_seed(unlabeled_logits, seed=seed, mode="feature")

            # supervised part
            one_hot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
            sup_loss = self._sup_criterion(label_logits.softmax(1), one_hot_target)
            # regularized part
            reg_loss = self.regularization(
                seed=seed,
                labeled_image=labeled_image,
                labeled_target=labeled_target,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf,
                unlabeled_tf_logits=unlabeled_tf_logits,
                unlabeled_logits_tf=unlabeled_logits_tf,
                label_group=unl_group,
                partition_group=unl_partition,
                labeled_filename=labeled_filename,
                unlabeled_filename=unlabeled_filename,
                affine_transformer=partial(self.transform_with_seed, seed=seed, mode="feature")
            )

        total_loss = sup_loss + reg_loss
        # gradient backpropagation
        self.scale_loss(total_loss).backward(retain_graph=retain_graph)
        self.optimizer_step(self._optimizer, cur_iter=cur_batch_num)

        # recording can be here or in the regularization method
        if self.on_master:
            with torch.no_grad():
                self.meters["sup_loss"].add(sup_loss.item())
                self.meters["sup_dice"].add(label_logits.max(1)[1], labeled_target.squeeze(1),
                                            group_name=label_group)
                self.meters["reg_loss"].add(reg_loss.item())

    def _forward_pass(self, labeled_image, unlabeled_image, unlabeled_image_tf):
        n_l, n_unl = len(labeled_image), len(unlabeled_image)
        if self._two_stage:
            label_logits = self._model(labeled_image)
            with self._bn_context(self._model):
                unlabeled_logits, unlabeled_tf_logits = \
                    torch.split(self._model(torch.cat([unlabeled_image, unlabeled_image_tf], dim=0)),
                                [n_unl, n_unl], dim=0)
            return label_logits, unlabeled_logits, unlabeled_tf_logits
        predict_logits = self._model(torch.cat([labeled_image, unlabeled_image, unlabeled_image_tf], dim=0))
        label_logits, unlabeled_logits, unlabeled_tf_logits = \
            torch.split(predict_logits, [n_l, n_unl, n_unl], dim=0)
        return label_logits, unlabeled_logits, unlabeled_tf_logits

    @property  # noqa
    @lru_cache()
    def _bn_context(self):
        return disable_tracking_bn_stats if self._disable_bn else nullcontext

    @staticmethod
    def _unzip_data(data, device):
        (image, target), (image_ct, target_ct), filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return (image, image_ct), target, filename, partition, group

    def _regularization(self, **kwargs):
        if len(self._hooks) > 0:
            reg_losses = [h(**kwargs) for h in self._hooks]
            return sum(reg_losses)
        return torch.tensor(0, device=self.device, dtype=torch.float)


class FineTuneEpocher(SemiSupervisedEpocher, ABC):
    meter_focus = "ft"

    def __init__(self, *, model: nn.Module, optimizer: optimizerType, labeled_loader: dataIterType,
                 sup_criterion: criterionType, num_batches: int, cur_epoch=0, device="cpu", scaler: GradScaler,
                 accumulate_iter: int, **kwargs) -> None:
        super().__init__(model=model, optimizer=optimizer, labeled_loader=labeled_loader, sup_criterion=sup_criterion,
                         num_batches=num_batches, cur_epoch=cur_epoch, device=device, scaler=scaler,
                         accumulate_iter=accumulate_iter, **kwargs)

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super().configure_meters(meters)
        meters.delete_meter("reg_loss")
        return meters

    def _forward_pass(self, labeled_image, **kwargs):
        return self._model(labeled_image)

    def _batch_update(self, *, cur_batch_num: int, labeled_image, labeled_target, label_group, retain_graph=False,
                      **kwargs):
        self.optimizer_zero(self._optimizer, cur_iter=cur_batch_num)
        # allowing to manipulate the gradient for after_batch_update_hook

        with self.autocast:
            label_logits: Tensor = self.forward_pass(labeled_image=labeled_image)  # noqa
            # supervised part
            onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
            sup_loss = self._sup_criterion(label_logits.softmax(1), onehot_target)

        total_loss = sup_loss
        # gradient backpropagation
        self.scale_loss(total_loss).backward(retain_graph=retain_graph)
        self.optimizer_step(self._optimizer, cur_iter=cur_batch_num)
        # recording can be here or in the regularization method
        if self.on_master:
            with torch.no_grad():
                self.meters["sup_loss"].add(sup_loss.item())
                self.meters["sup_dice"].add(label_logits.max(1)[1], labeled_target.squeeze(1),
                                            group_name=label_group)


class DMTEpcoher(SemiSupervisedEpocher):
    """
    This class verified the feasibility of `higher` package in the context of mean teacher
    """
    meter_focus = "dmt"

    def __init__(self, *, model: nn.Module, teacher_model: nn.Module, optimizer: optimizerType,
                 labeled_loader: SizedIterable, unlabeled_loader: SizedIterable, sup_criterion: criterionType,
                 num_batches: int, cur_epoch=0, device="cpu", two_stage: bool = False, disable_bn: bool = False,
                 scaler: GradScaler, accumulate_iter: int, mt_criterion: CriterionType, ema_updater: EMAUpdater,
                 mt_weight=10.0, meta_weight=0.001, **kwargs) -> None:
        super().__init__(model=model, optimizer=optimizer, labeled_loader=labeled_loader,
                         unlabeled_loader=unlabeled_loader, sup_criterion=sup_criterion, num_batches=num_batches,
                         cur_epoch=cur_epoch, device=device, two_stage=two_stage, disable_bn=disable_bn, scaler=scaler,
                         accumulate_iter=accumulate_iter, **kwargs)
        self._teacher_model = teacher_model
        self._mt_criterion = mt_criterion
        self._mt_weight = mt_weight
        self._ema_updater = ema_updater
        self._meta_weight = meta_weight

    def _assertion(self):
        assert len(self._hooks) == 0
        assert not self.scaler._enabled, "only support traditional training."  # noqa

    def _batch_update(self, *, cur_batch_num: int, labeled_image, labeled_target, labeled_filename, label_group,
                      unlabeled_image, unlabeled_image_tf, seed, unl_group, unl_partition, unlabeled_filename,
                      retain_graph=False, **kwargs):
        self.optimizer_zero(self._optimizer, cur_iter=cur_batch_num)
        label_logits, unlabeled_logits, unlabeled_tf_logits = self.forward_pass(
            labeled_image=labeled_image,
            unlabeled_image=unlabeled_image,
            unlabeled_image_tf=unlabeled_image_tf
        )

        unlabeled_logits_tf = self.transform_with_seed(unlabeled_logits, seed=seed)
        # supervised part
        one_hot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
        sup_loss = self._sup_criterion(label_logits.softmax(1), one_hot_target)
        # regularized part
        reg_loss = self.mt_update(
            teacher_model=self._teacher_model, unlabeled_tf_logits=unlabeled_tf_logits,
            unlabeled_image=unlabeled_image,
            affine_transformer=partial(self.transform_with_seed, seed=seed)
        )
        total_loss = sup_loss + reg_loss * 0.1
        self.teacher_model.zero_grad()
        old_teacher = deepcopy(self.teacher_model.state_dict())

        # gradient backpropagation
        first_deriv = torch.autograd.grad(total_loss, tuple(self._model.parameters()), create_graph=True)
        model_s_1 = [v - self._meta_weight * g for v, g in zip(self._model.parameters(), first_deriv)]
        model_t_1 = [0.999 * v_t.detach() + 0.001 * v_s_1 for v_t, v_s_1 in
                     zip(self.teacher_model.parameters(), model_s_1)]

        for p, p_ in zip(self.teacher_model.parameters(), model_t_1):
            setattr(p, "data", p_.data)
            setattr(p, "grad", p_.grad)

        meta_loss = self._sup_criterion(self._teacher_model(labeled_image).softmax(1), one_hot_target)

        meta_grad = torch.autograd.grad(meta_loss, tuple(self._model.parameters()), only_inputs=True)
        self._optimizer.step()

        # recording can be here or in the regularization method
        self.teacher_model.load_state_dict(old_teacher)
        self._ema_updater(ema_model=self.teacher_model, student_model=self._model)

        if self.on_master:
            with torch.no_grad():
                self.meters["sup_loss"].add(sup_loss.item())
                self.meters["sup_dice"].add(label_logits.max(1)[1], labeled_target.squeeze(1),
                                            group_name=label_group)
                self.meters["reg_loss"].add(reg_loss.item())

    def mt_update(self, *, teacher_model, unlabeled_tf_logits, unlabeled_image, affine_transformer):
        # taken from mean teacher
        student_unlabeled_tf_prob = unlabeled_tf_logits.softmax(1)
        with torch.no_grad():
            teacher_unlabeled_prob = teacher_model(unlabeled_image).softmax(1)
            teacher_unlabeled_prob_tf = affine_transformer(teacher_unlabeled_prob)
        return self._mt_criterion(teacher_unlabeled_prob_tf, student_unlabeled_tf_prob)

    @property
    def teacher_model(self):
        return self._teacher_model
