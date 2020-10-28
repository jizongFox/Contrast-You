import random

import torch

from contrastyou.arch.unet import freeze_grad
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation  # noqa
from contrastyou.epocher._utils import write_predict, write_img_target  # noqa
from contrastyou.helper import get_dataset
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.meters2 import EpochResultDict
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.type import T_loss
from semi_seg._utils import FeatureExtractor
from .base import PretrainEpocher
from .comparable import InfoNCEEpocher
from .._utils import ContrastiveProjectorWrapper


class _InfoNCEPretrainEpocher(PretrainEpocher):
    """
    this is to enable the total_freedom of the transformation
    """

    def init(self, *, chain_dataloader=None, **kwargs):
        super().init(chain_dataloader=chain_dataloader, **kwargs)
        assert self._feature_position == ["Conv5"]
        assert get_dataset(chain_dataloader)._transform._total_freedom is True  # noqa

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        self._model.train()
        assert self._model.training, self._model.training
        report_dict = EpochResultDict()
        with FeatureExtractor(self._model, self._feature_position) as self._fextractor:  # noqa
            for i, data in zip(self._indicator, self._chain_dataloader):
                seed = random.randint(0, int(1e7))
                image, image_tf, unlabeled_filename, unl_partition, unl_group = \
                    self._unzip_data(data, self._device)
                n_l, n_unl = 0, len(image)

                with FixRandomSeed(seed):
                    image_tf2 = torch.stack([self._affine_transformer(x) for x in image], dim=0)

                predict_logits = self._model(torch.cat([image_tf, image_tf2], dim=0))

                unlabel_logits, unlabel_tf_logits = torch.split(
                    predict_logits, [n_unl, n_unl], dim=0
                )

                with FixRandomSeed(seed):
                    unlabel_logits_tf = torch.stack([self._affine_transformer(x) for x in unlabel_logits], dim=0)

                assert unlabel_logits_tf.shape == unlabel_tf_logits.shape, (
                    unlabel_logits_tf.shape, unlabel_tf_logits.shape)

                # regularized part
                reg_loss = self.regularization(
                    unlabeled_tf_logits=unlabel_tf_logits,
                    unlabeled_logits_tf=unlabel_logits_tf,
                    seed=seed,
                    unlabeled_image=image,
                    unlabeled_image_tf=image_tf,
                    label_group=unl_group,
                    partition_group=unl_partition,
                    unlabeled_filename=unlabeled_filename,
                )
                total_loss = self._reg_weight * reg_loss
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

    @staticmethod
    def _unzip_data(data, device):
        (image, _), (image_tf, _), filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return image, image_tf, filename, partition, group


class InfoNCEPretrainEpocher(_InfoNCEPretrainEpocher, InfoNCEEpocher):

    def init(self, *, chain_dataloader=None, projectors_wrapper: ContrastiveProjectorWrapper = None,
             infoNCE_criterion: T_loss = None, **kwargs):
        super(InfoNCEPretrainEpocher, self).init(projectors_wrapper=projectors_wrapper,
                                                 infoNCE_criterion=infoNCE_criterion, chain_dataloader=chain_dataloader)

    def _run(self, *args, **kwargs) -> EpochResultDict:
        with freeze_grad(self._model, self._feature_position) as self._model:
            return super(InfoNCEPretrainEpocher, self)._run(*args, **kwargs)
