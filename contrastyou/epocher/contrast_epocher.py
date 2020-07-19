from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from deepclustering2 import optim
from deepclustering2.epoch import _Epocher, proxy_trainer  # noqa
from deepclustering2.loss import KL_div
from deepclustering2.meters2 import EpochResultDict, MeterInterface, AverageValueMeter, UniversalDice
from deepclustering2.tqdm import tqdm
from deepclustering2.trainer.trainer import T_loader, T_loss, T_optim
from deepclustering2.utils import simplex, class2one_hot, np
from ._utils import preprocess_input_with_twice_transformation, unfold_position


class PretrainEncoderEpoch(_Epocher):
    """using a pretrained network to train with a data loader with contrastive loss."""

    def __init__(self, model: nn.Module, projection_head: nn.Module, optimizer: optim.Optimizer,
                 pretrain_encoder_loader: T_loader = None, contrastive_criterion: T_loss = None, num_batches: int = 0,
                 cur_epoch=0, device="cpu", *args,
                 **kwargs) -> None:
        super().__init__(model, cur_epoch, device)
        self._projection_head = projection_head
        self._optimizer = optimizer
        self._pretrain_encoder_loader = pretrain_encoder_loader
        self._contrastive_criterion = contrastive_criterion
        self._num_batches = num_batches

    @classmethod
    def create_from_trainer(cls, trainer):
        # todo: complete the code here
        pass

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("contrastive_loss", AverageValueMeter())
        meters.register_meter("lr",AverageValueMeter())
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        assert self._model.training, self._model.training
        self.meters["lr"].add(self._optimizer.param_groups[0]["lr"])

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:  # noqa
            for i, data in zip(indicator, self._pretrain_encoder_loader):
                (img, _), (img_tf, _), filename, partition_list, group_list = self._preprocess_data(data, self._device)
                _, (e5, *_), *_ = self._model(torch.cat([img, img_tf], dim=0), return_features=True)
                global_enc, global_tf_enc = torch.chunk(F.normalize(self._projection_head(e5), dim=1), chunks=2, dim=0)
                # todo: convert representation to distance
                labels = self._mask_generation(partition_list, group_list)
                contrastive_loss = self._contrastive_criterion(torch.stack([global_enc, global_tf_enc], dim=1),
                                                               labels=labels)
                self._optimizer.zero_grad()
                contrastive_loss.backward()
                self._optimizer.step()
                # todo: meter recording.
                with torch.no_grad():
                    self.meters["contrastive_loss"].add(contrastive_loss.item())
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
        return report_dict

    @staticmethod
    def _preprocess_data(data, device):
        return preprocess_input_with_twice_transformation(data, device)

    @staticmethod
    def _mask_generation(partition_list: List[str], group_list: List[str]):
        """override this to provide more mask """
        return partition_list


class PretrainDecoderEpoch(PretrainEncoderEpoch):
    """using a pretrained network to train with a dataloader, for decoder part"""

    def __init__(self, model: nn.Module, projection_head: nn.Module, optimizer: optim.Optimizer,
                 pretrain_decoder_loader: T_loader = None, contrastive_criterion: T_loss = None, num_batches: int = 0,
                 cur_epoch=0, device="cpu", *args, **kwargs) -> None:
        super().__init__(model, projection_head, optimizer, pretrain_decoder_loader, contrastive_criterion, num_batches,
                         cur_epoch, device, *args, **kwargs)
        self._pretrain_decoder_loader = self._pretrain_encoder_loader

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        assert self._model.training, self._model.training
        self.meters["lr"].add(self._optimizer.param_groups[0]["lr"])

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:  # noqa
            for i, data in zip(indicator, self._pretrain_decoder_loader):
                (img, _), (img_tf, _), filename, partition_list, group_list = self._preprocess_data(data, self._device)
                _, *_, (_, d4, *_) = self._model(torch.cat([img, img_tf], dim=0), return_features=True)
                local_enc, local_tf_enc = torch.chunk(F.normalize(self._projection_head(d4), dim=1), chunks=2, dim=0)
                # todo: convert representation to distance
                local_enc_unfold, _ = unfold_position(local_enc, partition_num=(2, 2))
                local_tf_enc_unfold, _fold_partition = unfold_position(local_tf_enc, partition_num=(2, 2))
                b, *_ = local_enc_unfold.shape
                local_enc_unfold_norm = F.normalize(local_enc_unfold.view(b, -1), p=2, dim=1)
                local_tf_enc_unfold_norm = F.normalize(local_tf_enc_unfold.view(b, -1), p=2, dim=1)

                labels = self._mask_generation(partition_list, group_list, _fold_partition)
                contrastive_loss = self._contrastive_criterion(
                    torch.stack([local_enc_unfold_norm, local_tf_enc_unfold_norm], dim=1),
                    labels=labels
                )
                if torch.isnan(contrastive_loss):
                    raise RuntimeError()
                self._optimizer.zero_grad()
                contrastive_loss.backward()
                self._optimizer.step()
                # todo: meter recording.
                with torch.no_grad():
                    self.meters["contrastive_loss"].add(contrastive_loss.item())
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
        return report_dict

    @staticmethod
    def _mask_generation(partition_list: List[str], group_list: List[str], folder_partition: List[str]):
        if len(folder_partition) > len(partition_list):
            ratio = int(len(folder_partition) / len(partition_list))
            partition_list = partition_list.tolist() * ratio
            group_list = group_list * ratio

        def tolabel(encode):
            unique_labels = np.unique(encode)
            mapping = {k: i for i, k in enumerate(unique_labels)}
            return [mapping[k] for k in encode]

        return tolabel([str(g) + str(p) + str(f) for g, p, f in zip(group_list, partition_list, folder_partition)])


class FineTuneEpoch(_Epocher):
    def __init__(self, model: nn.Module, optimizer: T_optim, labeled_loader: T_loader, num_batches: int = 100,
                 cur_epoch=0, device="cpu") -> None:
        super().__init__(model, cur_epoch, device)
        assert isinstance(num_batches, int) and num_batches > 0, num_batches
        self._labeled_loader = labeled_loader
        self._sup_criterion = KL_div()
        self._num_batches = num_batches
        self._optimizer = optimizer

    @classmethod
    @proxy_trainer
    def create_from_trainer(cls, trainer):
        return cls(trainer._model, trainer._labeled_loader, trainer._unlabeled_loader, trainer._sup_criterion,
                   trainer._reg_criterion, trainer._num_batches, trainer._cur_epoch, trainer._device,
                   trainer._reg_weight)

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("ds", UniversalDice(4, [1, 2, 3]))
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        assert self._model.training, self._model.training
        report_dict: EpochResultDict
        self.meters["lr"].add(self._optimizer.param_groups[0]["lr"])
        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
            for i, label_data in zip(indicator, self._labeled_loader):
                (labelimage, labeltarget), _, filename, partition_list, group_list \
                    = self._preprocess_data(label_data, self._device)
                predict_logits = self._model(labelimage)
                assert not simplex(predict_logits), predict_logits

                onehot_ltarget = class2one_hot(labeltarget.squeeze(1), 4)
                sup_loss = self._sup_criterion(predict_logits.softmax(1), onehot_ltarget)

                self._optimizer.zero_grad()
                sup_loss.backward()
                self._optimizer.step()

                with torch.no_grad():
                    self.meters["sup_loss"].add(sup_loss.item())
                    self.meters["ds"].add(predict_logits.max(1)[1], labeltarget.squeeze(1),
                                          group_name=list(group_list))
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
            report_dict = self.meters.tracking_status()
        return report_dict

    @staticmethod
    def _preprocess_data(data, device):
        return preprocess_input_with_twice_transformation(data, device)
