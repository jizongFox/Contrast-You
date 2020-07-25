import random

import torch
from contrastyou.epocher._utils import unfold_position
from deepclustering2 import optim
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.meters2 import EpochResultDict
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.trainer.trainer import T_loader, T_loss
from torch import nn
from torch.nn import functional as F

from .contrast_epocher import PretrainDecoderEpoch as _PretrainDecoderEpoch
from .contrast_epocher import PretrainEncoderEpoch as _PretrainEncoderEpoch


class IICPretrainEcoderEpoch(_PretrainEncoderEpoch):

    def __init__(self, model: nn.Module, projection_head: nn.Module, projection_classifier: nn.Module,
                 optimizer: optim.Optimizer, pretrain_encoder_loader: T_loader,
                 contrastive_criterion: T_loss, num_batches: int = 0,
                 cur_epoch=0, device="cpu", group_option: str = "partition", iic_weight_ratio=1) -> None:
        """
        :param model:
        :param projection_head:
        :param projection_classifier: classification head
        :param optimizer:
        :param pretrain_encoder_loader: infinite dataloader with `total freedom = True`
        :param contrastive_criterion:
        :param num_batches:
        :param cur_epoch:
        :param device:
        :param iic_weight_ratio: iic weight_ratio
        """
        super(IICPretrainEcoderEpoch, self).__init__(model, projection_head, optimizer, pretrain_encoder_loader,
                                                     contrastive_criterion, num_batches,
                                                     cur_epoch, device, group_option=group_option)
        assert pretrain_encoder_loader is not None
        self._projection_classifier = projection_classifier
        from ..losses.iic_loss import IIDLoss
        self._iic_criterion = IIDLoss()
        self._iic_weight_ratio = iic_weight_ratio

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        assert self._model.training, self._model.training
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:  # noqa
            for i, data in zip(indicator, self._pretrain_encoder_loader):
                (img, _), (img_tf, _), filename, partition_list, group_list = self._preprocess_data(data, self._device)
                _, (e5, *_), *_ = self._model(torch.cat([img, img_tf], dim=0), return_features=True)
                global_enc, global_tf_enc = torch.chunk(F.normalize(self._projection_head(e5), dim=1), chunks=2, dim=0)
                global_probs, global_tf_probs = torch.chunk(self._projection_classifier(e5), chunks=2, dim=0)
                # fixme: here lack of some code for IIC
                labels = self._label_generation(partition_list, group_list)
                contrastive_loss = self._contrastive_criterion(torch.stack([global_enc, global_tf_enc], dim=1),
                                                               labels=labels)
                iic_loss = self._iic_criterion(global_probs, global_tf_probs)  # todo
                total_loss = self._iic_weight_ratio * iic_loss + (1 - self._iic_weight_ratio) * contrastive_loss
                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()
                # todo: meter recording.
                with torch.no_grad():
                    self.meters["contrastive_loss"].add(contrastive_loss.item())
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
        return report_dict


class IICPretrainDecoderEpoch(_PretrainDecoderEpoch):
    def __init__(self, model: nn.Module, projection_head: nn.Module, projection_classifier: nn.Module,
                 optimizer: optim.Optimizer, pretrain_decoder_loader: T_loader, contrastive_criterion: T_loss,
                 iic_criterion: T_loss, num_batches: int = 0, cur_epoch=0, device="cpu") -> None:
        super().__init__(model, projection_head, optimizer, pretrain_decoder_loader, contrastive_criterion, num_batches,
                         cur_epoch, device)
        self._projection_classifer = projection_classifier
        self._iic_criterion = iic_criterion

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        assert self._model.training, self._model.training
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:  # noqa
            for i, data in zip(indicator, self._pretrain_decoder_loader):
                (img, _), (img_ctf, _), filename, partition_list, group_list = self._preprocess_data(data, self._device)
                seed = random.randint(0, int(1e5))
                with FixRandomSeed(seed):
                    img_gtf = torch.stack([self._transformer(x) for x in img], dim=0)
                assert img_gtf.shape == img.shape, (img_gtf.shape, img.shape)

                _, *_, (_, d4, *_) = self._model(torch.cat([img_gtf, img_ctf], dim=0), return_features=True)
                d4_gtf, d4_ctf = torch.chunk(d4, chunks=2, dim=0)
                with FixRandomSeed(seed):
                    d4_ctf_gtf = torch.stack([self._transformer(x) for x in d4_ctf], dim=0)
                assert d4_ctf_gtf.shape == d4_ctf.shape, (d4_ctf_gtf.shape, d4_ctf.shape)
                d4_tf = torch.cat([d4_gtf, d4_ctf_gtf], dim=0)
                local_enc_tf, local_enc_tf_ctf = torch.chunk(self._projection_head(d4_tf), chunks=2, dim=0)
                # todo: convert representation to distance
                local_enc_unfold, _ = unfold_position(local_enc_tf, partition_num=(2, 2))
                local_tf_enc_unfold, _fold_partition = unfold_position(local_enc_tf_ctf, partition_num=(2, 2))
                b, *_ = local_enc_unfold.shape
                local_enc_unfold_norm = F.normalize(local_enc_unfold.view(b, -1), p=2, dim=1)
                local_tf_enc_unfold_norm = F.normalize(local_tf_enc_unfold.view(b, -1), p=2, dim=1)

                labels = self._label_generation(partition_list, group_list, _fold_partition)
                contrastive_loss = self._contrastive_criterion(
                    torch.stack([local_enc_unfold_norm, local_tf_enc_unfold_norm], dim=1),
                    labels=labels
                )
                if torch.isnan(contrastive_loss):
                    raise RuntimeError(contrastive_loss)
                self._optimizer.zero_grad()
                contrastive_loss.backward()
                self._optimizer.step()
                # todo: meter recording.
                with torch.no_grad():
                    self.meters["contrastive_loss"].add(contrastive_loss.item())
                    report_dict = self.meters.tracking_status()
                    indicator.set_postfix_dict(report_dict)
        return report_dict
