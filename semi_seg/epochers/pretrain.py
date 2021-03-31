import os
from pathlib import Path

import torch
from deepclustering2.configparser._utils import get_config  # noqa
from deepclustering2.decorator.decorator import _disable_tracking_bn_stats as disable_bn  # noqa
from torch import Tensor

from contrastyou.losses.contrast_loss2 import is_normalized
from contrastyou.losses.iic_loss import _ntuple  # noqa
from semi_seg.epochers import unl_extractor
from ._mixins import _PretrainEpocherMixin, _PretrainMonitorEpocherMxin
from .comparable import InfoNCEEpocher
from .miepocher import ConsistencyMIEpocher
from .miepocher import MITrainEpocher
from .newepocher import EncoderDenseContrastEpocher, EncoderDenseMixupContrastEpocher


# override batch loop in order to ignore the supervised loss.
class InfoNCEPretrainEpocher(_PretrainEpocherMixin, InfoNCEEpocher):
    pass


class InfoNCEPretrainMonitorEpocher(_PretrainMonitorEpocherMxin, InfoNCEEpocher):
    def _global_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                        label_group):
        """methods go for global vectors"""
        assert len(proj_tf_feature.shape) == 2, proj_tf_feature.shape
        assert is_normalized(proj_feature_tf) and is_normalized(proj_feature_tf)
        config = get_config(scope="base")
        save_dir = os.path.join("runs", config["Trainer"]["save_dir"])
        tag = f"features/{self.trainer._cur_epoch}"  # noqa
        if tag:
            tag = Path(save_dir) / tag
            tag.mkdir(parents=True, exist_ok=True)
            torch.save(proj_tf_feature.cpu().detach(), str(tag / label_group[0]))

    def _regularization(self, *, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, label_group,
                        partition_group, **kwargs):
        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2

        _ = [
            self.generate_infonce(
                feature_name=n, features=f, projector=p, seed=seed, partition_group=partition_group,
                label_group=label_group) for n, f, p in
            zip(self._fextractor.feature_names, unl_extractor(self._fextractor, n_uls=n_uls), self._projectors_wrapper)
        ]

        return torch.tensor(0, device=self.device)


class MIPretrainEpocher(_PretrainEpocherMixin, MITrainEpocher):
    pass


class UDAIICPretrainEpocher(_PretrainEpocherMixin, ConsistencyMIEpocher):
    pass


class ExperimentalPretrainEpocher(_PretrainEpocherMixin, EncoderDenseContrastEpocher):
    pass


class ExperimentalPretrainMixinEpocher(_PretrainEpocherMixin, EncoderDenseMixupContrastEpocher):
    pass
