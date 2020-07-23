import torch
from deepclustering2 import optim
from deepclustering2.meters2 import EpochResultDict
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.trainer.trainer import T_loader, T_loss
from torch import nn
from torch.nn import functional as F

from .contrast_epocher import PretrainEncoderEpoch as _PretrainEncoderEpoch


class IICPretrainEcoderEpoch(_PretrainEncoderEpoch):

    def __init__(self, model: nn.Module, projection_head: nn.Module, projection_classifier: nn.Module,
                 optimizer: optim.Optimizer, pretrain_encoder_loader: T_loader = None,
                 contrastive_criterion: T_loss = None, num_batches: int = 0,
                 cur_epoch=0, device="cpu", iic_weight_ratio=1, *args, **kwargs) -> None:
        """

        :param model:
        :param projection_head: here the projection head should be a classifier
        :param optimizer:
        :param pretrain_encoder_loader:
        :param contrastive_criterion:
        :param num_batches:
        :param cur_epoch:
        :param device:
        :param args:
        :param kwargs:
        """
        assert pretrain_encoder_loader is not None
        self._projection_classifier = projection_classifier
        from ..losses.iic_loss import IIDLoss
        self._iic_criterion = IIDLoss()
        self._iic_weight_ratio = iic_weight_ratio
        super().__init__(model, projection_head, optimizer, pretrain_encoder_loader, contrastive_criterion, num_batches,
                         cur_epoch, device, *args, **kwargs)

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.train()
        assert self._model.training, self._model.training
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:  # noqa
            for i, data in zip(indicator, self._pretrain_encoder_loader):
                (img, _), (img_tf, _), filename, partition_list, group_list = self._preprocess_data(data, self._device)
                _, (e5, *_), *_ = self._model(torch.cat([img, img_tf], dim=0), return_features=True)
                global_enc, global_tf_enc = torch.chunk(F.normalize(self._projection_head(e5), dim=1), chunks=2, dim=0)
                # fixme: here lack of some code for IIC
                labels = self._label_generation(partition_list, group_list)
                contrastive_loss = self._contrastive_criterion(torch.stack([global_enc, global_tf_enc], dim=1),
                                                               labels=labels)
                iic_loss = self._iic_criterion()  # todo
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
