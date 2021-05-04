import random
from abc import abstractmethod
from contextlib import nullcontext
from typing import Callable, Dict

import torch
from deepclustering2.augment.tensor_augment import TensorRandomFlip
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats  # noqa
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.type import T_loader, T_loss, T_optim
from deepclustering2.utils import class2one_hot, ExceptionIgnorer
from loguru import logger
from torch import nn, Tensor
from torch.utils.data.dataloader import DataLoader

from contrastyou.epochers.base import _Epocher  # noqa
from contrastyou.meters import AverageValueMeter, UniversalDice, MeterInterface, SurfaceMeter
from semi_seg.epochers.helper import preprocess_input_with_single_transformation, write_predict, write_img_target, \
    preprocess_input_with_twice_transformation
from semi_seg.utils import _num_class_mixin

EpochResultDict = Dict[str, Dict[str, float]]


# to enable init and _init, in order to insert assertion of params
class Epocher(_num_class_mixin, _Epocher):
    _forward_pass: Callable

    def init(self, **kwargs):
        super(Epocher, self).init(**kwargs)
        self.assert_()

    @abstractmethod
    def _init(self, *args, **kwargs):
        pass

    def forward_pass(self, *args, **kwargs):
        return self._forward_pass(*args, **kwargs)

    def assert_(self):
        pass


# ======== validation epochers =============
class EvalEpocher(Epocher):

    def __init__(self, *, model: nn.Module, loader: T_loader, sup_criterion: T_loss, cur_epoch=0, device="cpu") \
        -> None:
        assert isinstance(loader, DataLoader), \
            f"val_loader should be an instance of DataLoader, given {loader.__class__.__name__}."
        super().__init__(model=model, num_batches=len(loader), cur_epoch=cur_epoch, device=device)
        self._loader = loader
        self._sup_criterion = sup_criterion

    def _init(self, **kwargs):
        pass

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_classes = list(range(1, C))
        meters.register_meter("loss", AverageValueMeter())
        meters.register_meter("dice", UniversalDice(C, report_classes=report_classes))
        return meters

    def get_score(self):
        return self.meters["dice"].summary()["DSC_mean"]

    def _batch_optimization(self, *, val_img, val_target, file_path, partition, scan, **kwargs):
        val_logits = self._model(val_img)
        onehot_target = class2one_hot(val_target.squeeze(1), self.num_classes)

        val_loss = self._sup_criterion(val_logits.softmax(1), onehot_target, disable_assert=True)
        self.meters["loss"].add(val_loss.item())
        self.meters["dice"].add(val_logits.max(1)[1], val_target.squeeze(1), scan_name=scan)
        return val_loss

    @torch.no_grad()
    def _run(self, *args, **kwargs):
        self._model.eval()
        for i, val_data in zip(self.indicator, self._loader):
            val_img, val_target, file_path, partition, group = self._unzip_data(val_data, self._device)
            val_loss = self.batch_optimization(val_img=val_img, val_target=val_target, file_path=file_path,
                                               partition=partition, scan=group)

            report_dict = self.meters.statistics()
            self.indicator.set_postfix_statics(report_dict)

    @staticmethod
    def _unzip_data(data, device):
        image, target, filename, partition, group = preprocess_input_with_single_transformation(data, device)
        return image, target, filename, partition, group


class InferenceEpocher(EvalEpocher):

    def _init(self, *, save_dir: str):
        self._save_dir = save_dir  # noqa

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super().configure_meters(meters)
        C = self.num_classes
        report_classes = list(range(1, C))
        meters.register_meter("hd", SurfaceMeter(C=C, report_classes=report_classes, meter_name="hausdorff"))
        return meters

    def _batch_optimization(self, *, val_img, val_target, file_path, partition, scan, **kwargs):
        val_logits = self._model(val_img)

        write_img_target(val_img, val_target, self._save_dir, file_path)
        write_predict(val_logits, self._save_dir, file_path, )
        onehot_target = class2one_hot(val_target.squeeze(1), self.num_classes)

        val_loss = self._sup_criterion(val_logits.softmax(1), onehot_target, disable_assert=True)

        self.meters["loss"].add(val_loss.item())
        self.meters["dice"].add(val_logits.max(1)[1], val_target.squeeze(1), scan_name=scan)
        with ExceptionIgnorer(RuntimeError):
            self.meters["hd"].add(val_logits.max(1)[1], val_target.squeeze(1))
        return val_loss


# ========= base training epocher ===============
class TrainEpocher(Epocher):

    def get_score(self):
        raise RuntimeError()

    def __init__(self, *, model: nn.Module, optimizer: T_optim, labeled_loader: T_loader, unlabeled_loader: T_loader,
                 sup_criterion: T_loss, num_batches: int, cur_epoch=0, device="cpu", train_with_two_stage: bool = False,
                 disable_reg_bn_track: bool = False, **kwargs) -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device)

        self._optimizer = optimizer
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._sup_criterion = sup_criterion
        self._affine_transformer = TensorRandomFlip(axis=[1, 2], threshold=0.8)

        self.train_with_two_stage = train_with_two_stage  # highlight: this is the parameter to use two stage training
        logger.opt(depth=1).trace("{} set to be using {} stage training", self.__class__.__name__,
                                  "two" if self.train_with_two_stage else "single")
        self._disable_bn = disable_reg_bn_track  # highlight: disable the bn accumulation
        if self._disable_bn:
            logger.trace("{} set to disable bn tracking", self.__class__.__name__)

    def _init(self, *, reg_weight: float, **kwargs):
        pass

    def configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(TrainEpocher, self).configure_meters(meters)
        C = self.num_classes
        report_axis = list(range(1, C))

        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("sup_dice", UniversalDice(C, report_classes=report_axis))

        return meters

    def _run(self, *args, **kwargs):
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        assert self._model.training, self._model.training
        return self._run_semi(*args, **kwargs)

    def assert_(self):
        labeled_dataset = self._labeled_loader._dataset  # noqa
        lab_transform = labeled_dataset._transforms  # noqa
        assert not lab_transform._total_freedom

        if self._unlabeled_loader is not None:
            unlabeled_dataset = self._unlabeled_loader.dataset
            unlab_transform = unlabeled_dataset._transforms  # noqa
            assert not unlab_transform._total_freedom

    def _batch_optimization(self, *, labeled_image, labeled_target, unlabeled_image, unlabeled_image_tf, seed,
                            unl_group, unl_partition, unlabeled_filename, labeled_filename, label_group):

        label_logits, unlabel_logits, unlabel_tf_logits = self.forward_pass(
            labeled_image=labeled_image,
            unlabeled_image=unlabeled_image,
            unlabeled_image_tf=unlabeled_image_tf
        )

        with FixRandomSeed(seed):
            unlabel_logits_tf = torch.stack([self._affine_transformer(x) for x in unlabel_logits], dim=0)

        onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
        sup_loss = self._sup_criterion(label_logits.softmax(1), onehot_target)
        # regularized part
        reg_loss = self.regularization(
            unlabeled_tf_logits=unlabel_tf_logits,
            unlabeled_logits_tf=unlabel_logits_tf,
            seed=seed,
            unlabeled_image=unlabeled_image,
            unlabeled_image_tf=unlabeled_image_tf,
            label_group=unl_group,
            partition_group=unl_partition,
            unlabeled_filename=unlabeled_filename,
            labeled_filename=labeled_filename
        )

        total_loss = sup_loss + reg_loss

        with torch.no_grad():
            self.meters["sup_loss"].add(sup_loss.item())
            self.meters["sup_dice"].add(label_logits.max(1)[1], labeled_target.squeeze(1),
                                        scan_name=label_group)
            self.meters["reg_loss"].add(reg_loss.item())
        return total_loss

    def _run_semi(self, *args, **kwargs):
        self._model.train()
        for self.cur_batch_num, labeled_data, unlabeled_data in zip(self.indicator, self._labeled_loader,
                                                                    self._unlabeled_loader):

            (labeled_image, _), labeled_target, labeled_filename, _, label_group = \
                self._unzip_data(labeled_data, self._device)
            (unlabeled_image, unlabeled_image_ct), _unlabeled_target, unlabeled_filename, unl_partition, unl_group \
                = self._unzip_data(unlabeled_data, self._device)

            seed = random.randint(0, int(1e7))
            with FixRandomSeed(seed):
                unlabeled_image_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_image_ct], dim=0)

            total_loss = self.batch_optimization(
                labeled_image=labeled_image,
                labeled_target=labeled_target,
                unlabeled_image=unlabeled_image,
                unlabeled_image_tf=unlabeled_image_tf,
                seed=seed,
                unl_group=unl_group,
                unl_partition=unl_partition,
                unlabeled_filename=unlabeled_filename,
                labeled_filename=labeled_filename,
                label_group=label_group
            )

            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()

            # recording can be here or in the regularization method
            if self.on_master():
                report_dict = self.meters.statistics()
                self.indicator.set_postfix_statics(report_dict)

    def _forward_pass(self, labeled_image, unlabeled_image, unlabeled_image_tf):
        n_l, n_unl = len(labeled_image), len(unlabeled_image)
        if not self.train_with_two_stage:
            # if train with only single stage
            predict_logits = self._model(torch.cat([labeled_image, unlabeled_image, unlabeled_image_tf], dim=0))
            label_logits, unlabel_logits, unlabel_tf_logits = torch.split(predict_logits,
                                                                          [n_l, n_unl, n_unl], dim=0)
        else:
            # train with two stages, while their feature extractions are the same
            label_logits = self._model(labeled_image)
            bn_context = _disable_tracking_bn_stats if self._disable_bn else nullcontext

            with bn_context(self._model):
                unlabel_logits, unlabel_tf_logits = torch.split(
                    self._model(torch.cat([unlabeled_image, unlabeled_image_tf], dim=0)),
                    [n_unl, n_unl],
                    dim=0
                )
        return label_logits, unlabel_logits, unlabel_tf_logits

    @staticmethod
    def _unzip_data(data, device):
        (image, target), (image_ct, target_ct), filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        assert torch.allclose(target, target_ct)
        return (image, image_ct), target, filename, partition, group

    def regularization(self, **kwargs):
        return self._regularization(**kwargs)

    def _regularization(self, **kwargs):
        return torch.tensor(0, dtype=torch.float, device=self._device)


class FineTuneEpocher(TrainEpocher, ):

    def __init__(self, *, model: nn.Module, optimizer: T_optim, labeled_loader: T_loader,
                 sup_criterion: T_loss, num_batches: int, cur_epoch=0, device="cpu", **kwargs) -> None:
        super().__init__(model=model, optimizer=optimizer, labeled_loader=labeled_loader,
                         unlabeled_loader=None, sup_criterion=sup_criterion, num_batches=num_batches,  # noqa
                         cur_epoch=cur_epoch, device=device, train_with_two_stage=False,
                         disable_reg_bn_track=False)

    def _run(self, *args, **kwargs):
        self._model.train()
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer))
        return self._run_only_label(*args, **kwargs)

    def _batch_optimization(self, *, labeled_image, labeled_target, label_group, **kwargs):
        label_logits: Tensor = self.forward_pass(labeled_image=labeled_image)  # noqa
        # supervised part
        onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
        sup_loss = self._sup_criterion(label_logits.softmax(1), onehot_target)
        with torch.no_grad():
            self.meters["sup_loss"].add(sup_loss.item())
            self.meters["sup_dice"].add(label_logits.max(1)[1], labeled_target.squeeze(1),
                                        scan_name=label_group)
        return sup_loss

    def _run_only_label(self, *args, **kwargs):
        for self.cur_batch_num, labeled_data in zip(self.indicator, self._labeled_loader):
            (labeled_image, _), labeled_target, labeled_filename, _, label_group = \
                self._unzip_data(labeled_data, self._device)

            total_loss = self.batch_optimization(
                labeled_image=labeled_image, labeled_target=labeled_target, label_group=label_group)

            # gradient backpropagation
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()
            # recording can be here or in the regularization method
            if self.on_master():
                report_dict = self.meters.statistics()
                self.indicator.set_postfix_statics(report_dict)

    def _forward_pass(self, labeled_image, **kwargs):
        label_logits = self._model(labeled_image)
        return label_logits
