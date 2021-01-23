import random
from typing import List

import torch
from torch import nn

from contrastyou.arch.unet import freeze_grad
from contrastyou.featextractor.unet import FeatureExtractorWithIndex as FeatureExtractor
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.meters2 import EpochResultDict, MeterInterface
from deepclustering2.optim import get_lrs_from_optimizer
from .comparable import InfoNCEEpocher
from .miepocher import MITrainEpocher, ConsistencyMIEpocher


# ======== base pretrain epocher mixin ================
class _PretrainEpocherMixin:
    """
    PretrainEpocher makes all images goes to regularization, permitting to use the other classes to create more pretrain
    models
    """

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meter = super()._configure_meters(meters)
        meter.delete_meters(["sup_loss", "sup_dice", "reg_weight"])
        return meter

    def init(self, *, chain_dataloader, **kwargs):
        # extend the interface for original class with chain_dataloader
        super().init(**kwargs)
        self._chain_dataloader = chain_dataloader  # noqa

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        assert self._model.training, self._model.training

        with FeatureExtractor(self._model, self._feature_position) as self._fextractor:  # noqa
            return self._run_pretrain(*args, **kwargs)

    def _run_pretrain(self, *args, **kwargs):
        report_dict = EpochResultDict()
        for i, data in zip(self._indicator, self._chain_dataloader):
            seed = random.randint(0, int(1e7))
            unlabeled_image, unlabeled_target, unlabeled_filename, unl_partition, unl_group = \
                self._unzip_data(data, self._device)
            n_l, n_unl = 0, len(unlabeled_image)

            with FixRandomSeed(seed):
                unlabeled_image_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_image], dim=0)
            assert unlabeled_image_tf.shape == unlabeled_image.shape, \
                (unlabeled_image_tf.shape, unlabeled_image.shape)

            # clear feature cache
            self._fextractor.clear()

            predict_logits = self._model(torch.cat([unlabeled_image, unlabeled_image_tf], dim=0))

            unlabel_logits, unlabel_tf_logits = torch.split(predict_logits, [n_unl, n_unl], dim=0)

            with FixRandomSeed(seed):
                unlabel_logits_tf = torch.stack([self._affine_transformer(x) for x in unlabel_logits], dim=0)

            assert unlabel_logits_tf.shape == unlabel_tf_logits.shape, (
                unlabel_logits_tf.shape, unlabel_tf_logits.shape)

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
            )
            total_loss = reg_loss
            # gradient backpropagation
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()
            # recording can be here or in the regularization method
            if self.on_master():
                with torch.no_grad():
                    self.meters["reg_loss"].add(reg_loss.item())
                    report_dict = self.meters.tracking_status()
                    self._indicator.set_postfix_dict(report_dict)
        return report_dict


class __freeze_grad_mixin:
    _model: nn.Module
    _feature_position: List[str]

    def _run(self, *args, **kwargs):
        with freeze_grad(self._model, self._feature_position) as self._model:  # noqa
            return super()._run(*args, **kwargs)


class InfoNCEPretrainEpocher(__freeze_grad_mixin, _PretrainEpocherMixin, InfoNCEEpocher):
    pass


class MIPretrainEpocher(__freeze_grad_mixin, _PretrainEpocherMixin, MITrainEpocher):
    pass


class UDAIICPretrainEpocher(__freeze_grad_mixin, _PretrainEpocherMixin, ConsistencyMIEpocher):
    pass


from .newepocher import NewEpocher


class ExperimentalPretrainEpocher(__freeze_grad_mixin, _PretrainEpocherMixin, NewEpocher):
    pass
