import random
from abc import ABC
from contextlib import nullcontext
from functools import lru_cache, partial
from typing import Any, Dict, Optional, final

import torch
from loguru import logger
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from contrastyou.augment.tensor_augment import TensorRandomFlip
from contrastyou.epochers.base import EpocherBase as _EpocherBase
from contrastyou.losses.kl import KL_div
from contrastyou.meters import MeterInterface, UniversalDice, AverageValueMeter
from contrastyou.types import criterionType, optimizerType, dataIterType
from contrastyou.utils import get_dataset, class_name, fix_all_seed_for_transforms, get_lrs_from_optimizer
from contrastyou.utils.general import class2one_hot
from contrastyou.utils.utils import disable_tracking_bn_stats, get_model
from semi_seg.epochers.helper import preprocess_input_with_twice_transformation, \
    preprocess_input_with_single_transformation
from semi_seg.helper import SizedIterable


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
        self._initialized = False

    @final
    def run(self):
        if not self._initialized:
            raise RuntimeError(f"Call {class_name(self)}.init() before {class_name(self)}.run()")
        return super(EpocherBase, self).run()

    def init(self):
        """we added an assertion to control the hyper-parameters."""
        self._assertion()
        self._initialized = True

    def _batch_update(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        returning the predicted dictionary, to pass to possible metric meter.
        """
        ...

    def batch_update(self, **kwargs) -> Optional[Dict[str, Any]]:
        """batch updater given labeled and unlabeled images, including model update, zero grad and optimizer steps.
        Gan style update should be included in this function as well."""
        for h in self._hooks:
            h.before_batch_update(**kwargs)
        predicted_result_dict = self._batch_update(**kwargs)
        for h in self._hooks:
            h.after_batch_update(**kwargs, result_dict=predicted_result_dict)
        return predicted_result_dict

    def _assertion(self):
        pass

    def forward_pass(self, **kwargs):
        for h in self._hooks:
            h.before_forward_pass(**kwargs)
        result = self._forward_pass(**kwargs)
        for h in self._hooks:
            h.after_forward_pass(**kwargs, result_dict=result)
        return result

    def _forward_pass(self, **kwargs) -> Any:
        ...

    def regularization(self, **kwargs):
        for h in self._hooks:
            h.before_regularization(**kwargs)
        result = self._regularization(**kwargs)
        for h in self._hooks:
            h.after_regularization(**kwargs, result_dict=result)
        return result

    def _regularization(self, **kwargs):
        return torch.tensor(0, dtype=torch.float, device=self._device)


class EvalEpocher(EpocherBase):
    meter_focus = "eval"

    def get_score(self) -> float:
        metric = self.meters["dice"].summary()
        return metric["DSC_mean"]

    def __init__(self, *, model: nn.Module, loader: DataLoader, sup_criterion: KL_div, cur_epoch=0, device="cpu",
                 scaler: GradScaler, accumulate_iter: int) -> None:
        super().__init__(model=model, num_batches=len(loader), cur_epoch=cur_epoch, device=device, scaler=scaler,
                         accumulate_iter=accumulate_iter)
        self._loader = loader
        self._sup_criterion: KL_div = sup_criterion

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("loss", AverageValueMeter())
        meters.register_meter("dice", UniversalDice(C, report_axises=report_axis))
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
                              eval_group=group)

            report_dict = self.meters.statistics()
            self.indicator.set_postfix_statics(report_dict)

    def _batch_update(self, *, eval_img, eval_target, eval_group):
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


class SemiSupervisedEpocher(EpocherBase, ABC):
    meter_focus = "semi"

    def _assertion(self):
        assert_transform_freedom(self._labeled_loader, False)
        if self._unlabeled_loader is not None:
            assert_transform_freedom(self._unlabeled_loader, False)

    def __init__(self, *, model: nn.Module, optimizer: optimizerType, labeled_loader: dataIterType,
                 unlabeled_loader: dataIterType, sup_criterion: criterionType, num_batches: int, cur_epoch=0,
                 device="cpu", two_stage: bool = False, disable_bn: bool = False, scaler: GradScaler,
                 accumulate_iter: int, **kwargs) -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device, scaler=scaler,
                         accumulate_iter=accumulate_iter)
        self._optimizer = optimizer
        self._labeled_loader: SizedIterable = labeled_loader
        self._unlabeled_loader: SizedIterable = unlabeled_loader
        self._sup_criterion = sup_criterion
        self._affine_transformer = TensorRandomFlip(axis=[1, 2], threshold=0.8)
        self._two_stage = two_stage
        logger.opt(depth=1).trace("{} set to be using {} stage training", self.__class__.__name__,
                                  "two" if self._two_stage else "single")
        self._disable_bn = disable_bn
        if self._disable_bn:
            logger.debug("{} set to disable bn tracking", self.__class__.__name__)

    def transform_with_seed(self, features, seed):
        with fix_all_seed_for_transforms(seed):
            features_tf = torch.stack([self._affine_transformer(x) for x in features], dim=0)
        return features_tf

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(SemiSupervisedEpocher, self).configure_meters(meters)
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("sup_dice", UniversalDice(C, report_axises=report_axis))
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

            unlabeled_image_tf = self.transform_with_seed(unlabeled_image_cf, seed=seed)

            self.batch_update(cur_batch_num=self.cur_batch_num,
                              labeled_image=labeled_image,
                              labeled_target=labeled_target,
                              labeled_filename=labeled_filename,
                              label_group=label_group, unlabeled_image=unlabeled_image,
                              unlabeled_image_tf=unlabeled_image_tf,
                              seed=seed, unl_group=unl_group, unl_partition=unl_partition,
                              unlabeled_filename=unlabeled_filename)

            report_dict = self.meters.statistics()
            self.indicator.set_postfix_statics(report_dict, cache_time=20)

    def _batch_update(self, *, cur_batch_num: int, labeled_image, labeled_target, labeled_filename, label_group,
                      unlabeled_image, unlabeled_image_tf, seed, unl_group, unl_partition, unlabeled_filename,
                      **kwargs):
        with self.autocast:
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
            reg_loss = self.regularization(
                seed=seed,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf,
                unlabeled_tf_logits=unlabeled_tf_logits,
                unlabeled_logits_tf=unlabeled_logits_tf,
                label_group=unl_group,
                partition_group=unl_partition,
                labeled_filename=labeled_filename,
                unlabeled_filename=unlabeled_filename,
                affine_transformer=partial(self.transform_with_seed, seed=seed)
            )

        total_loss = sup_loss + reg_loss
        # gradient backpropagation
        self.scale_loss(total_loss).backward()
        self.optimizer_step(self._optimizer, cur_iter=cur_batch_num)
        self.optimizer_zero(self._optimizer, cur_iter=cur_batch_num)

        # recording can be here or in the regularization method
        if self.on_master():
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
        label_logits = self._model(labeled_image)
        return label_logits

    def _batch_update(self, *, cur_batch_num: int, labeled_image, labeled_target, label_group, **kwargs):
        with self.autocast:
            label_logits: Tensor = self.forward_pass(labeled_image=labeled_image)  # noqa
            # supervised part
            onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
            sup_loss = self._sup_criterion(label_logits.softmax(1), onehot_target)

        total_loss = sup_loss
        # gradient backpropagation
        self.scale_loss(total_loss).backward()
        self.optimizer_step(self._optimizer, cur_iter=cur_batch_num)
        self.optimizer_zero(self._optimizer, cur_iter=cur_batch_num)
        # recording can be here or in the regularization method
        if self.on_master():
            with torch.no_grad():
                self.meters["sup_loss"].add(sup_loss.item())
                self.meters["sup_dice"].add(label_logits.max(1)[1], labeled_target.squeeze(1),
                                            group_name=label_group)
