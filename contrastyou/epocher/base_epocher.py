from typing import Union, Tuple

import torch
from torch.utils.data import DataLoader

from deepclustering2 import ModelMode
from deepclustering2.epoch import _Epocher, proxy_trainer  # noqa
from deepclustering2.loss import simplex
from deepclustering2.meters2 import EpochResultDict, MeterInterface, AverageValueMeter, UniversalDice
from deepclustering2.models import Model
from deepclustering2.tqdm import tqdm
from deepclustering2.trainer.trainer import T_loader, T_loss
from deepclustering2.utils import class2one_hot
from ._utils import preprocess_input_with_once_transformation, preprocess_input_train_fs, \
    preprocess_input_with_twice_transformation


class FSEpocher:
    class TrainEpoch(_Epocher):

        def __init__(self, model: Model, data_loader: T_loader, sup_criteiron: T_loss, num_batches: int = 100,
                     cur_epoch=0, device="cpu") -> None:
            super().__init__(model, cur_epoch, device)
            self._data_loader = data_loader
            self._sup_criterion = sup_criteiron
            self._num_batches = num_batches

        @classmethod
        @proxy_trainer
        def create_from_trainer(cls, trainer):
            return cls(trainer._model, trainer._tra_loader, trainer._sup_criterion, trainer._num_batches,  # noqa
                       trainer._cur_epoch, trainer._device)  # noqa

        def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
            meters.register_meter("lr", AverageValueMeter())
            meters.register_meter("sup_loss", AverageValueMeter())
            meters.register_meter("ds", UniversalDice(4, [1, 2, 3]))
            return meters

        def _run(self, *args, **kwargs) -> Union[EpochResultDict, Tuple[EpochResultDict, float]]:
            self._model.set_mode(ModelMode.TRAIN)
            assert self._model.training, self._model.training
            self.meters["lr"].add(self._model.get_lr()[0])
            with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:  # noqa
                for i, data in zip(indicator, self._data_loader):
                    images, targets, filename, partition_list, group_list = self._preprocess_data(data, self._device)
                    predict_logits = self._model(images)
                    assert not simplex(predict_logits), predict_logits.shape
                    onehot_targets = class2one_hot(targets.squeeze(1), 4)
                    loss = self._sup_criterion(predict_logits.softmax(1), onehot_targets)
                    self._model.zero_grad()
                    loss.backward()
                    self._model.step()
                    with torch.no_grad():
                        self.meters["sup_loss"].add(loss.item())
                        self.meters["ds"].add(predict_logits.max(1)[1], targets.squeeze(1), group_name=list(group_list))
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
            report_dict = self.meters.tracking_status()
            return report_dict

        @staticmethod
        def _preprocess_data(data, device):
            return preprocess_input_train_fs(data, device)

    class EvalEpoch(TrainEpoch):
        def __init__(self, model: Model, val_data_loader: T_loader, sup_criterion, cur_epoch=0, device="cpu"):
            super().__init__(model=model, data_loader=val_data_loader, sup_criteiron=sup_criterion,
                             num_batches=None, cur_epoch=cur_epoch, device=device)  # noqa
            assert isinstance(val_data_loader, DataLoader), type(val_data_loader)

        @classmethod
        @proxy_trainer
        def create_from_trainer(cls, trainer):
            return cls(trainer._model, trainer._val_loader, trainer._sup_criterion, trainer._cur_epoch,  # noqa
                       trainer._device)  # noqa

        def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
            super()._configure_meters(meters)
            meters.delete_meters(["lr"])
            return meters

        @torch.no_grad()
        def _run(self, *args, **kwargs) -> Union[EpochResultDict, Tuple[EpochResultDict, float]]:
            self._model.set_mode(ModelMode.EVAL)
            assert not self._model.training, self._model.training
            with tqdm(range(len(self._data_loader))).set_desc_from_epocher(self) as indicator:
                for i, data in zip(indicator, self._data_loader):
                    images, targets, filename, partiton_list, group_list = self._preprocess_data(data, self._device)
                    predict_logits = self._model(images)
                    assert not simplex(predict_logits), predict_logits.shape
                    onehot_targets = class2one_hot(targets.squeeze(1), 4)
                    loss = self._sup_criterion(predict_logits.softmax(1), onehot_targets, disable_assert=True)
                    self.meters["sup_loss"].add(loss.item())
                    self.meters["ds"].add(predict_logits.max(1)[1], targets.squeeze(1), group_name=list(group_list))
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
            report_dict = self.meters.tracking_status()
            return report_dict, report_dict["ds"]["DSC_mean"]

        @staticmethod
        def _preprocess_data(data, device):
            return preprocess_input_with_once_transformation(data, device)


class SemiEpocher:
    class TrainEpoch(_Epocher):

        def __init__(self, model: Model, labeled_loader: T_loader, unlabeled_loader: T_loader, sup_criteiron: T_loss,
                     reg_criterion: T_loss, num_batches: int = 100, cur_epoch=0, device="cpu",
                     reg_weight: float = 0.001) -> None:
            super().__init__(model, cur_epoch, device)
            assert isinstance(num_batches, int) and num_batches > 0, num_batches
            self._labeled_loader = labeled_loader
            self._unlabeled_loader = unlabeled_loader
            self._sup_criterion = sup_criteiron
            self._reg_criterion = reg_criterion
            self._num_batches = num_batches
            self._reg_weight = reg_weight

        @classmethod
        @proxy_trainer
        def create_from_trainer(cls, trainer):
            return cls(trainer._model, trainer._labeled_loader, trainer._unlabeled_loader, trainer._sup_criterion,
                       trainer._reg_criterion, trainer._num_batches, trainer._cur_epoch, trainer._device,
                       trainer._reg_weight)

        def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
            meters.register_meter("lr", AverageValueMeter())
            meters.register_meter("sup_loss", AverageValueMeter())
            meters.register_meter("reg_weight", AverageValueMeter())
            meters.register_meter("reg_loss", AverageValueMeter())
            meters.register_meter("ds", UniversalDice(4, [1, 2, 3]))
            return meters

        def _run(self, *args, **kwargs) -> EpochResultDict:
            self._model.set_mode(ModelMode.TRAIN)
            assert self._model.training, self._model.training
            report_dict: EpochResultDict
            self.meters["lr"].add(self._model.get_lr()[0])
            self.meters["reg_weight"].add(self._reg_weight)

            with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
                for i, label_data, unlabel_data in zip(indicator, self._labeled_loader, self._unlabeled_loader):
                    (labelimage, labeltarget), (labelimage_tf, labeltarget_tf), filename, partition_list, group_list, (
                        unlabelimage, unlabelimage_tf) = self._preprocess_data(label_data, unlabel_data, self._device)
                    predict_logits = self._model(
                        torch.cat([labelimage, labelimage_tf, unlabelimage, unlabelimage_tf], dim=0),
                        force_simplex=False)
                    assert not simplex(predict_logits), predict_logits
                    label_logit, label_logit_tf, unlabel_logit, unlabel_logit_tf \
                        = torch.split(predict_logits,
                                      [len(labelimage), len(labelimage_tf), len(unlabelimage), len(unlabelimage_tf)],
                                      dim=0)
                    onehot_ltarget = class2one_hot(torch.cat([labeltarget.squeeze(), labeltarget_tf.squeeze()], dim=0),
                                                   4)
                    sup_loss = self._sup_criterion(torch.cat([label_logit, label_logit_tf], dim=0).softmax(1),
                                                   onehot_ltarget)
                    reg_loss = self._reg_criterion(unlabel_logit.softmax(1), unlabel_logit_tf.softmax(1))
                    total_loss = sup_loss + reg_loss * self._reg_weight

                    self._model.zero_grad()
                    total_loss.backward()
                    self._model.step()

                    with torch.no_grad():
                        self.meters["sup_loss"].add(sup_loss.item())
                        self.meters["ds"].add(label_logit.max(1)[1], labeltarget.squeeze(1),
                                              group_name=list(group_list))
                        self.meters["reg_loss"].add(reg_loss.item())
                        report_dict = self.meters.tracking_status()
                        indicator.set_postfix_dict(report_dict)
                report_dict = self.meters.tracking_status()
            return report_dict

        @staticmethod
        def _preprocess_data(labeled_input, unlabeled_input, device):
            return preprocess_input_with_twice_transformation(labeled_input, unlabeled_input, device)

    class EvalEpoch(FSEpocher.EvalEpoch):
        pass

    

