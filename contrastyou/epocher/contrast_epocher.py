import torch

from deepclustering2 import ModelMode
from deepclustering2.epoch import _Epocher, proxy_trainer  # noqa
from deepclustering2.meters2 import EpochResultDict
from deepclustering2.models import Model
from deepclustering2.tqdm import tqdm
from deepclustering2.trainer.trainer import T_loader, T_loss
from ._utils import preprocess_input_with_twice_transformation


class PretrainEncoderEpoch(_Epocher):

    def __init__(self, model: Model, cur_epoch=0, device="cpu", pretrain_encoder_loader: T_loader = None,
                 contrastive_criterion: T_loss = None, num_batches: int = 0, *args, **kwargs) -> None:
        # check if the network has enable_grad for encoder and decoder
        assert hasattr(model._torchnet, "disable_grad_encoder") and \
               hasattr(model._torchnet, "enable_grad_encoder"), model._torchnet
        # check if pretrain_encoder_loader is with infinitesampler
        from deepclustering2.dataloader.sampler import _InfiniteRandomIterator
        assert isinstance(pretrain_encoder_loader._sampler_iter, _InfiniteRandomIterator), pretrain_encoder_loader
        assert isinstance(num_batches, int) and num_batches >= 1, num_batches
        super().__init__(model, cur_epoch, device)
        self._pretrain_encoder_loader = pretrain_encoder_loader
        self._contrastive_criterion = contrastive_criterion
        self._num_batches = num_batches

    @classmethod
    def create_from_trainer(cls, trainer):
        # todo: complete the code here
        pass

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.set_mode(ModelMode.TRAIN)
        assert self._model.training, self._model.training
        self._model._torchnet.enable_grad_encoder()  # noqa
        self._model._torchnet.disable_grad_decoder()  # noqa

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
            for i, data in zip(indicator, self._pretrain_encoder_loader):
                (img, _), (img_tf, _), filename, partition_list, group_list = self._preprocess_data(data, self._device)
                representations = self._model(torch.cat([img, img_tf], dim=0))
                img_repr, img_tf_repr = torch.chunk(representations, chunks=2, dim=0)
                # todo: convert representation to distance
                contrastive_loss = self._contrastive_criterion(img_repr, img_tf_repr, partition_list, group_list)
                self._model.zero_grad()
                contrastive_loss.backward()
                self._model.step()
                # todo: meter recording.
        # todo return EpochResult.
        return None

    @staticmethod
    def _preprocess_data(data, device):
        return preprocess_input_with_twice_transformation(data, device)


class PretrainDecoderEpoch(PretrainEncoderEpoch):

    def __init__(self, model: Model, cur_epoch=0, device="cpu", pretrain_decoder_loader: T_loader = None,
                 contrastive_criterion: T_loss = None, num_batches: int = 0, *args, **kwargs) -> None:
        super().__init__(model, cur_epoch, device, pretrain_decoder_loader, contrastive_criterion, num_batches, *args,
                         **kwargs)
        self._pretrain_decoder_loader = self._pretrain_encoder_loader

    @classmethod
    def create_from_trainer(cls, trainer):
        super().create_from_trainer(trainer)

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self._model.set_mode(ModelMode.TRAIN)
        assert self._model.training, self._model.training
        self._model._torchnet.enable_grad_decoder()  # noqa
        self._model._torchnet.disable_grad_encoder()  # noqa

        with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
            for i, data in zip(indicator, self._pretrain_decoder_loader):
                (img, _), (img_tf, _), filename, partition_list, group_list = self._preprocess_data(data, self._device)
                representations = self._model(torch.cat([img, img_tf], dim=0))
                img_repr, img_tf_repr = torch.chunk(representations, chunks=2, dim=0)
                # todo: convert representation to distance
                contrastive_loss = self._contrastive_criterion(img_repr, img_tf_repr, partition_list, group_list)
                self._model.zero_grad()
                contrastive_loss.backward()
                self._model.step()
                # todo: meter recording.
        # todo return EpochResult.
        return None

# class FineTuneEpoch(_Epocher):
#     def __init__(self, model: Model, labeled_loader: T_loader, unlabeled_loader: T_loader, sup_criteiron: T_loss,
#                  reg_criterion: T_loss, num_batches: int = 100, cur_epoch=0, device="cpu",
#                  reg_weight: float = 0.001) -> None:
#         super().__init__(model, cur_epoch, device)
#         assert isinstance(num_batches, int) and num_batches > 0, num_batches
#         self._labeled_loader = labeled_loader
#         self._unlabeled_loader = unlabeled_loader
#         self._sup_criterion = sup_criteiron
#         self._reg_criterion = reg_criterion
#         self._num_batches = num_batches
#         self._reg_weight = reg_weight
#
#     @classmethod
#     @proxy_trainer
#     def create_from_trainer(cls, trainer):
#         return cls(trainer._model, trainer._labeled_loader, trainer._unlabeled_loader, trainer._sup_criterion,
#                    trainer._reg_criterion, trainer._num_batches, trainer._cur_epoch, trainer._device,
#                    trainer._reg_weight)
#
#     def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
#         meters.register_meter("lr", AverageValueMeter())
#         meters.register_meter("sup_loss", AverageValueMeter())
#         meters.register_meter("reg_weight", AverageValueMeter())
#         meters.register_meter("reg_loss", AverageValueMeter())
#         meters.register_meter("ds", UniversalDice(4, [1, 2, 3]))
#         return meters
#
#     def _run(self, *args, **kwargs) -> EpochResultDict:
#         self._model.set_mode(ModelMode.TRAIN)
#         assert self._model.training, self._model.training
#         report_dict: EpochResultDict
#         self.meters["lr"].add(self._model.get_lr()[0])
#         self.meters["reg_weight"].add(self._reg_weight)
#
#         with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
#             for i, label_data, unlabel_data in zip(indicator, self._labeled_loader, self._unlabeled_loader):
#                 (labelimage, labeltarget), (labelimage_tf, labeltarget_tf), filename, partition_list, group_list, (
#                     unlabelimage, unlabelimage_tf) = self._preprocess_data(label_data, unlabel_data, self._device)
#                 predict_logits = self._model(
#                     torch.cat([labelimage, labelimage_tf, unlabelimage, unlabelimage_tf], dim=0),
#                     force_simplex=False)
#                 assert not simplex(predict_logits), predict_logits
#                 label_logit, label_logit_tf, unlabel_logit, unlabel_logit_tf \
#                     = torch.split(predict_logits,
#                                   [len(labelimage), len(labelimage_tf), len(unlabelimage), len(unlabelimage_tf)],
#                                   dim=0)
#                 onehot_ltarget = class2one_hot(torch.cat([labeltarget.squeeze(), labeltarget_tf.squeeze()], dim=0),
#                                                4)
#                 sup_loss = self._sup_criterion(torch.cat([label_logit, label_logit_tf], dim=0).softmax(1),
#                                                onehot_ltarget)
#                 reg_loss = self._reg_criterion(unlabel_logit.softmax(1), unlabel_logit_tf.softmax(1))
#                 total_loss = sup_loss + reg_loss * self._reg_weight
#
#                 self._model.zero_grad()
#                 total_loss.backward()
#                 self._model.step()
#
#                 with torch.no_grad():
#                     self.meters["sup_loss"].add(sup_loss.item())
#                     self.meters["ds"].add(label_logit.max(1)[1], labeltarget.squeeze(1),
#                                           group_name=list(group_list))
#                     self.meters["reg_loss"].add(reg_loss.item())
#                     report_dict = self.meters.tracking_status()
#                     indicator.set_postfix_dict(report_dict)
#             report_dict = self.meters.tracking_status()
#         return report_dict
#
#     @staticmethod
#     def _preprocess_data(labeled_input, unlabeled_input, device):
#         return preprocess_input_with_twice_transformation(labeled_input, unlabeled_input, device)
#
#
# class TrainEpoch(SemiEpocher.TrainEpoch):
#
#     def __init__(self, model: Model, labeled_loader: T_loader, unlabeled_loader: T_loader, sup_criteiron: T_loss,
#                  reg_criterion: T_loss, num_batches: int = 100, cur_epoch=0, device="cpu",
#                  reg_weight: float = 0.001) -> None:
#         super().__init__(model, labeled_loader, unlabeled_loader, sup_criteiron, reg_criterion, num_batches, cur_epoch,
#                          device, reg_weight)
#         assert reg_criterion  # todo: add constraints on the reg_criterion
#
#     def _run(self, *args, **kwargs) -> EpochResultDict:
#         self._model.set_mode(ModelMode.TRAIN)
#         assert self._model.training, self._model.training
#         report_dict: EpochResultDict
#         self.meters["lr"].add(self._model.get_lr()[0])
#         self.meters["reg_weight"].add(self._reg_weight)
#
#         with tqdm(range(self._num_batches)).set_desc_from_epocher(self) as indicator:
#             for i, label_data, unlabel_data in zip(indicator, self._labeled_loader, self._unlabeled_loader):
#                 (labelimage, labeltarget), (labelimage_tf, labeltarget_tf), filename, partition_list, group_list, (
#                     unlabelimage, unlabelimage_tf) = self._preprocess_data(label_data, unlabel_data, self._device)
#                 predict_logits = self._model(
#                     torch.cat([labelimage, labelimage_tf, unlabelimage, unlabelimage_tf], dim=0),
#                     force_simplex=False, return_features=True)
#                 assert not simplex(predict_logits), predict_logits
#                 label_logit, label_logit_tf, unlabel_logit, unlabel_logit_tf \
#                     = torch.split(
#                     predict_logits,
#                     [len(labelimage), len(labelimage_tf), len(unlabelimage), len(unlabelimage_tf)],
#                     dim=0)
#                 onehot_ltarget = class2one_hot(torch.cat([labeltarget.squeeze(), labeltarget_tf.squeeze()], dim=0),
#                                                4)
#                 sup_loss = self._sup_criterion(torch.cat([label_logit, label_logit_tf], dim=0).softmax(1),
#                                                onehot_ltarget)
#                 reg_loss = self._reg_criterion(unlabel_logit.softmax(1), unlabel_logit_tf.softmax(1))
#                 total_loss = sup_loss + reg_loss * self._reg_weight
#
#                 self._model.zero_grad()
#                 total_loss.backward()
#                 self._model.step()
#
#                 with torch.no_grad():
#                     self.meters["sup_loss"].add(sup_loss.item())
#                     self.meters["ds"].add(label_logit.max(1)[1], labeltarget.squeeze(1),
#                                           group_name=list(group_list))
#                     self.meters["reg_loss"].add(reg_loss.item())
#                     report_dict = self.meters.tracking_status()
#                     indicator.set_postfix_dict(report_dict)
#             report_dict = self.meters.tracking_status()
#         return report_dict
