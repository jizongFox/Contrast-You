import warnings
from functools import lru_cache
from itertools import chain
from typing import Tuple, Optional, Dict

import torch
from sklearn.cluster import KMeans
from torch import Tensor
from torch.nn import functional as F

from contrastyou.arch.unet import UNet
from contrastyou.epocher._utils import unfold_position  # noqa
from contrastyou.helper import weighted_average_iter
from contrastyou.projectors.heads import ProjectionHead, LocalProjectionHead
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.meters2 import MeterInterface, AverageValueMeter
from deepclustering2.type import T_loss
from semi_seg._utils import ContrastiveProjectorWrapper as PrototypeProjectorWrapper
from .base import TrainEpocher
from .helper import unl_extractor


class PrototypeEpocher(TrainEpocher):
    only_with_labeled_data = False

    def init(self, *, reg_weight: float, prototype_projector: PrototypeProjectorWrapper = None, feature_buffers=None,
             infoNCE_criterion: T_loss = None, **kwargs):
        """
        :param reg_weight:  regularization weight
        :param prototype_projector: prototype projector to logits
        :param feature_buffers: buffered feature maps for each feature position and each slides
        :param infoNCE_criterion: T_loss for infoNCE.
        :param kwargs:
        :return:
        """
        assert prototype_projector is not None, prototype_projector
        assert infoNCE_criterion is not None, infoNCE_criterion
        super().init(reg_weight=reg_weight, **kwargs)
        self._projectors_wrapper = prototype_projector  # noqa
        self._feature_buffers: Optional[Dict[str, Dict[str, Tensor]]] = feature_buffers  # noqa
        self._infonce_criterion = infoNCE_criterion  # noqa
        self._kmeans_collection = dict()  # noqa
        warnings.warn(
            f"{self.__class__.__name__} now only support Conv5 for Encoder Descriptor, for simplification",
            category=RuntimeWarning
        )
        self._feature_position = ["Conv5"]
        self._feature_importance = [1.0, ]

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        meters.register_meter("mi", AverageValueMeter())
        return meters

    def run_kmeans(self, feature_collection: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :arg feature_collection: input feature collection
        :return: Center and labels
        """
        if len(feature_collection) == 0:
            raise NotImplementedError()
        for feature_position, sub_feature_collection in feature_collection.items():
            _kmeans = KMeans(n_clusters=100, n_jobs=10)
            self._kmeans_collection[feature_position] = _kmeans
            _features = list(sub_feature_collection.values())
            _processed_features = torch.cat(list(map(lambda x: x.view(x.size(0), -1).transpose(1, 0), _features)),
                                            dim=0)
            perm = torch.randperm(_processed_features.size(0))
            idx = perm[:100000]
            samples = _processed_features[idx]
            _kmeans.fit(samples)

    @property
    def cluster_center(self):
        if len(self._kmeans_collection) == 0:
            return None
        return {k: v.cluster_centers_ for k, v in self._kmeans_collection.items()}

    def get_kmeans(self, feature_position):
        return self._kmeans_collection[feature_position]

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, label_group,
                       partition_group, unlabeled_filename, labeled_filename, *args, **kwargs):

        feature_names = self._fextractor._feature_names  # noqa
        n_uls = len(unlabeled_tf_logits) * 2

        for feature_name, features in zip(feature_names, self._fextractor):
            for filename, _feature in zip(chain(labeled_filename, unlabeled_filename), features):
                if feature_name not in self._feature_buffers:
                    self._feature_buffers[feature_name] = dict()
                self._feature_buffers[feature_name][filename] = _feature.detach().cpu()

        def generate_infonce(features, projector):
            unlabeled_features, unlabeled_tf_features = torch.chunk(features, 2, dim=0)
            with FixRandomSeed(seed):
                unlabeled_features_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_features], dim=0)
            assert unlabeled_tf_features.shape == unlabeled_tf_features.shape, \
                (unlabeled_tf_features.shape, unlabeled_tf_features.shape)
            proj_tf_feature, proj_feature_tf = torch.chunk(
                projector(torch.cat([unlabeled_tf_features, unlabeled_features_tf], dim=0)), 2, dim=0
            )
            # normalization and label generation goes differently here.
            if isinstance(projector, ProjectionHead):
                norm_tf_feature, norm_feature_tf = F.normalize(proj_tf_feature, p=2, dim=1), \
                                                   F.normalize(proj_feature_tf, p=2, dim=1)
                assert len(norm_tf_feature.shape) == 2, norm_tf_feature.shape
                labels = self.global_label_generator(partition_list=partition_group, patient_list=label_group)
            elif isinstance(projector, LocalProjectionHead):
                proj_feature_tf_unfold, positional_label = unfold_position(proj_feature_tf,
                                                                           partition_num=proj_feature_tf.shape[-2:])
                proj_tf_feature_unfold, _ = unfold_position(proj_tf_feature, partition_num=proj_tf_feature.shape[-2:])

                __b = proj_tf_feature_unfold.size(0)

                labels = self.local_label_generator(partition_list=partition_group, patient_list=label_group,
                                                    location_list=positional_label)
                norm_feature_tf = F.normalize(proj_feature_tf_unfold.view(__b, -1), dim=1, p=2)
                norm_tf_feature = F.normalize(proj_tf_feature_unfold.view(__b, -1), dim=1, p=2)

            else:
                raise NotImplementedError(type(projector))
            return self._infonce_criterion(torch.stack([norm_feature_tf, norm_tf_feature], dim=1), labels=labels)

        losses = [generate_infonce(f, p) for f, p in
                  zip(unl_extractor(self._fextractor, n_uls=n_uls),
                      self._projectors_wrapper)
                  ]

        reg_loss = weighted_average_iter(losses, self._feature_importance)
        # self.meters["mi"].add(-reg_loss.item())
        # self.meters["individual_mis"].add(**dict(zip(
        #     self._feature_position,
        #     [-x.item() for x in losses]
        # )))
        return reg_loss

    @property  # noqa
    @lru_cache()
    def global_label_generator(self):
        from contrastyou.epocher._utils import GlobalLabelGenerator  # noqa
        return GlobalLabelGenerator()

    @property  # noqa
    @lru_cache()
    def local_label_generator(self):
        from contrastyou.epocher._utils import LocalLabelGenerator  # noqa
        return LocalLabelGenerator()


class DifferentiablePrototypeEpocher(TrainEpocher):
    """This Epocher is only for updating the encoder descriptor, using different ways"""

    def init(self, *, reg_weight: float, prototype_vectors: Tensor = None, prototype_nums=100, **kwargs):
        super().init(reg_weight=reg_weight, **kwargs)
        assert self._feature_position == [
            "Conv5"], f"Only support Conv5 for current simplification, given {','.join(self._feature_position)}"
        dim = UNet.dimension_dict[self._feature_position[0]]
        if prototype_vectors is None:
            self._prototype_vectors = torch.randn(prototype_nums, dim)  # noqa
        assert self._prototype_vectors.shape[1] == dim, self._prototype_vectors.shape

        assert self._prototype_vectors.requires_grad, self._prototype_vectors.requires_grad
        self._prototype_optimizer = torch.optim.Adam(  # noqa
            (self._prototype_vectors,), lr=1e-3, weight_decay=1e-4)  # todo: change parameter latter

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(DifferentiablePrototypeEpocher, self)._configure_meters(meters)
        meters.add("prototype_mi", AverageValueMeter())
        return meters

    def regularization(self, unlabeled_tf_logits: Tensor, unlabeled_logits_tf: Tensor, seed: int, label_group,
                       partition_group, unlabeled_filename, labeled_filename, *args, **kwargs):
        for n_feature, feature in enumerate(self._fextractor):
            pass

    def pairwise_distance(self, feature_map: Tensor):
        pass
