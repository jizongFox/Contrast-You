import torch
from torch import nn

from deepclustering2.meters2 import EpochResultDict
from deepclustering2.tqdm import tqdm
from deepclustering2.trainer.trainer import T_loader, T_loss
from deepclustering2.utils import simplex, class2one_hot
from .base_epocher import SemiEpocher


class TrainEpoch(SemiEpocher.TrainEpoch):

    def __init__(self, model: nn.Module, labeled_loader: T_loader, unlabeled_loader: T_loader, sup_criteiron: T_loss,
                 reg_criterion: T_loss, num_batches: int = 100, cur_epoch=0, device="cpu",
                 reg_weight: float = 0.001) -> None:
        super().__init__(model, labeled_loader, unlabeled_loader, sup_criteiron, reg_criterion, num_batches, cur_epoch,
                         device, reg_weight)
        assert reg_criterion  # todo: add constraints on the reg_criterion

    def _run(self, *args, **kwargs) -> EpochResultDict:
        assert self._model.training, self._model.training
        report_dict: EpochResultDict
        self.meters["lr"].add(self._model.get_lr()[0])
        self.meters["reg_weight"].add(self._reg_weight)

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
            for i, label_data, unlabel_data in zip(indicator, self._labeled_loader, self._unlabeled_loader):
                ((labelimage, labeltarget), (labelimage_tf, labeltarget_tf)), \
                filename, partition_list, group_list = self._preprocess_data(label_data, self._device)
                ((unlabelimage, _), (unlabelimage_tf, _)), unlabel_filename, \
                unlabel_partition_list, unlabel_group_list = self._preprocess_data(unlabel_data, self._device)

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
