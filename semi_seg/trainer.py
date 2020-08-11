from typing import Union, Tuple

import torch
from deepclustering2.epoch import _Epocher
from deepclustering2.meters2 import EpochResultDict, StorageIncomeDict, AverageValueMeter, UniversalDice
from deepclustering2.meters2 import MeterInterface
from deepclustering2.models import Model
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.tqdm import tqdm
from deepclustering2.trainer import Trainer
from deepclustering2.trainer.trainer import T_loader, T_loss, T_optim
from deepclustering2.utils import class2one_hot
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader

from contrastyou.augment import AffineTensorTransform
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation, \
    preprocess_input_with_single_transformation
from semi_seg._utils import FeatureExtractor


class EvalEpocher(_Epocher):

    def __init__(self, model: Union[Model, nn.Module], val_loader: T_loader, sup_criterion: T_loss, cur_epoch=0,
                 device="cpu") -> None:
        super().__init__(model, cur_epoch, device)
        assert isinstance(val_loader, DataLoader), \
            f"val_loader should be an instance of DataLoader, given {val_loader.__class__.__name__}."
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = 4
        report_axis = [1, 2, 3]
        meters.register_meter("loss", AverageValueMeter())
        meters.register_meter("dice", UniversalDice(C, report_axises=report_axis, ))
        return meters

    @torch.no_grad()
    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        self._model.eval()
        assert self._model.training is False, self._model.training
        with tqdm(self._val_loader).set_desc_from_epocher(self) as indicator:
            for i, val_data in enumerate(indicator):
                val_img, val_target, file_path, _, group = self._unzip_data(val_data, self._device)
                val_logits = self._model(val_img)
                onehot_target = class2one_hot(val_target.squeeze(1), 4)

                val_loss = self._sup_criterion(val_logits.softmax(1), onehot_target, disable_assert=True)

                self.meters["loss"].add(val_loss.item())
                self.meters["dice"].add(val_logits.max(1)[1], val_target.squeeze(1), group_name=group)

                report_dict = self.meters.tracking_status()
                indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters["dice"].summary()["DSC_mean"]

    @staticmethod
    def _unzip_data(data, device):
        image, target, filename, partition, group = preprocess_input_with_single_transformation(data, device)
        return image, target, filename, partition, group


class TrainEpocher(_Epocher):

    def __init__(self, model: Union[Model, nn.Module], optimizer: T_optim, labeled_loader: T_loader,
                 unlabeled_loader: T_loader, sup_criterion: T_loss, reg_weight: float, num_batches: int, cur_epoch=0,
                 device="cpu", feature_position=["Conv5", "Up_conv3"]) -> None:
        super().__init__(model, cur_epoch, device)
        self._optimizer = optimizer
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._sup_criterion = sup_criterion
        self._num_batches = num_batches
        self._reg_weight = reg_weight
        self._affine_transformer = AffineTensorTransform()
        self._affine_transformer_nearest = AffineTensorTransform(mode="nearest")
        self._feature_position = feature_position

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = 4
        report_axis = [1, 2, 3]
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("reg_loss", AverageValueMeter())
        meters.register_meter("sup_dice", UniversalDice(C, report_axises=report_axis, ))
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        self._model.train()
        assert self._model.training, self._model.training
        report_dict = {}
        with FeatureExtractor(self._model, self._feature_position) as self._fextractor:
            with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
                for i, labeled_data, unlabeled_data in zip(indicator, self._labeled_loader, self._unlabeled_loader):
                    labeled_image, labeled_target, labeled_filename, _, label_group = self._unzip_data(labeled_data,
                                                                                                       self._device)
                    unlabeled_image, *_ = self._unzip_data(unlabeled_data, self._device)
                    unlabeled_image_tf, AffineMatrix = self._affine_transformer(unlabeled_image, independent=True)

                    predict_logits = self._model(torch.cat([labeled_image, unlabeled_image, unlabeled_image_tf], dim=0))
                    label_logits, unlabel_logits, unlabel_tf_logits = \
                        torch.split(
                            predict_logits,
                            [len(labeled_image), len(unlabeled_image), len(unlabeled_image_tf)],
                            dim=0
                        )
                    unlabel_logits_tf, _ = self._affine_transformer(unlabel_logits, AffineMatrix)
                    # supervised part
                    onehot_target = class2one_hot(labeled_target.squeeze(1), 4)
                    sup_loss = self._sup_criterion(label_logits.softmax(1), onehot_target)
                    # regularized part
                    reg_loss = self.regularization(
                        unlabeled_tf_logits=unlabel_tf_logits,
                        unlabeled_logits_tf=unlabel_logits_tf,
                        affine_matrix=AffineMatrix
                    )
                    total_loss = sup_loss + self._reg_weight * reg_loss
                    # gradient backpropagation
                    self._optimizer.zero_grad()
                    total_loss.backward()
                    self._optimizer.step()
                    # recording can be here or in the regularization method
                    with torch.no_grad():
                        self.meters["sup_loss"].add(sup_loss.item())
                        self.meters["sup_dice"].add(label_logits.max(1)[1], labeled_target.squeeze(1),
                                                    group_name=label_group)
                        self.meters["reg_loss"].add(reg_loss.item())
                        report_dict = self.meters.tracking_status()
                        indicator.set_postfix_dict(report_dict)
        return report_dict

    @staticmethod
    def _unzip_data(data, device):
        (image, target), _, filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return image, target, filename, partition, group

    def regularization(self, *args, **kwargs):
        return torch.tensor(0, dtype=torch.float, device=self._device)


class UDATrainEpocher(TrainEpocher):

    def __init__(self, model: Union[Model, nn.Module], optimizer: T_optim, labeled_loader: T_loader,
                 unlabeled_loader: T_loader, sup_criterion: T_loss, reg_criterion: T_loss, reg_weight: float,
                 num_batches: float, cur_epoch: int = 0, device="cpu", feature_position=["Conv5", "Up_conv3"]) -> None:
        super().__init__(model, optimizer, labeled_loader, unlabeled_loader, sup_criterion, reg_weight, num_batches,
                         cur_epoch, device, feature_position)
        self._reg_criterion = reg_criterion

    def regularization(
        self,
        unlabeled_tf_logits: Tensor,
        unlabeled_logits_tf: Tensor,
        affine_matrix, *args, **kwargs
    ):
        return self._reg_criterion(
            unlabeled_tf_logits.softmax(1),
            unlabeled_logits_tf.softmax(1)
        )


class IICTrainEpocher(TrainEpocher):

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("mi", AverageValueMeter())
        return meters

    def __init__(self, model: Union[Model, nn.Module], projectors_wrapper: nn.Module, optimizer: nn.Module,
                 labeled_loader: T_optim, unlabeled_loader: T_loader, sup_criterion: T_loader, IIDSegCriterion: T_loss,
                 reg_weight: float, num_batches: int, cur_epoch: float = 0, device: int = "cpu",
                 feature_position=["Up_conv3", "Up_conv2"]) -> None:
        super().__init__(model, optimizer, labeled_loader, unlabeled_loader, sup_criterion, reg_weight, num_batches,
                         cur_epoch, device, feature_position)
        self._projectors_wrapper = projectors_wrapper
        self._IIDSegCriterion = IIDSegCriterion

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, affine_matrix, *args, **kwargs):
        # todo: adding projectors here.
        feature_names = self._fextractor._feature_names
        for f_name in feature_names:
            features = self._fextractor[f_name]
            unlabeled_length = len(unlabeled_tf_logits) * 2
            unlabeled_features = features[len(features) - unlabeled_length:]

            unlabeled_features, unlabeled_tf_features = torch.chunk(unlabeled_features, 2, dim=0)
            unlabeled_features_tf, _ = self._affine_transformer(unlabeled_features, affine_matrix)
        return torch.tensor(0, dtype=torch.float, device=self._device)


class UDAIICEpocher(IICTrainEpocher):
    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("uda_loss", AverageValueMeter())
        return meters

    def regularization(self):
        pass


class SemiTrainer(Trainer):

    def __init__(self, model: Union[Model, nn.Module], labeled_loader: T_loader, unlabeled_loader: T_loader,
                 val_loader: T_loader, sup_criterion: T_loss, save_dir: str = "base", max_epoch: int = 100,
                 num_batches: int = 100, device: str = "cpu", configuration=None):
        super().__init__(model, save_dir, max_epoch, num_batches, device, configuration)
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._val_loader = val_loader
        self._register_buffer("_best_score", 0)

    def _start_training(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            train_result: EpochResultDict
            eval_result: EpochResultDict
            cur_score: float
            train_result = self.run_epoch()
            with torch.no_grad():
                eval_result, cur_score = self.eval_epoch()
            # update lr_scheduler
            self._model.schedulerStep()
            storage_per_epoch = StorageIncomeDict(tra=train_result, val=eval_result)
            self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
            for k, v in storage_per_epoch.__dict__.items():
                self._writer.add_scalar_with_tag(k, v, global_step=self._cur_epoch)
            # save_checkpoint
            self.save(cur_score)
            # save storage result on csv file.
            self._storage.to_csv(self._save_dir)

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        assert self._model.training is True

        with tqdm(range(self._num_batch)) as indicator:
            for i, labeled_data, unlabeled_data in zip(indicator, self._labeled_loader, self._unlabeled_loader):
                pass