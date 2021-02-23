import torch
from deepclustering2.configparser._utils import get_config  # noqa
from deepclustering2.meters2 import MeterInterface, AverageValueMeter
from deepclustering2.type import T_loss
from torch import Tensor

from contrastyou.losses.contrast_loss import SupConLoss3, is_normalized
from contrastyou.losses.iic_loss import _ntuple  # noqa
from contrastyou.projectors.nn import Normalize
from .comparable import InfoNCEEpocher
from .._utils import ContrastiveProjectorWrapper


class ProposedEpocher3(InfoNCEEpocher):

    def _init(self, *, reg_weight: float, projectors_wrapper: ContrastiveProjectorWrapper = None,
              infoNCE_criterion: T_loss = None, **kwargs):
        super()._init(reg_weight=reg_weight, projectors_wrapper=projectors_wrapper, infoNCE_criterion=infoNCE_criterion,
                      **kwargs)
        self._infonce_criterion2 = SupConLoss3()

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super(ProposedEpocher3, self)._configure_meters(meters)
        config = get_config(scope="base")
        for f in config["ProjectorParams"]["GlobalParams"]["feature_names"]:
            m = "global"
            meters.register_meter(f"{f}_{m}/origin_mi", AverageValueMeter(), )
            meters.register_meter(f"{f}_{m}/weight_mi", AverageValueMeter(), )
        for f in config["ProjectorParams"]["DenseParams"]["feature_names"]:
            m = "dense"
            meters.register_meter(f"{f}_{m}/origin_mi", AverageValueMeter(), )
            meters.register_meter(f"{f}_{m}/weight_mi", AverageValueMeter(), )
        return meters

    def _global_infonce(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                        label_group) -> Tensor:
        unregularized_loss = super()._global_infonce(feature_name=feature_name, proj_tf_feature=proj_tf_feature,
                                                     proj_feature_tf=proj_feature_tf, partition_group=partition_group,
                                                     label_group=label_group)
        labels = self.global_label_generator(self.__encoder_method_name__)(partition_list=partition_group,
                                                                           patient_list=label_group)
        # labels = torch.Tensor(labels).to(device=self.device)
        # o_mask = torch.eq(labels[..., None], labels[None, ...]).float()
        sim_coef = self.generate_similarity_masks(proj_tf_feature, proj_feature_tf)
        self._infonce_criterion2: SupConLoss3
        regularized_loss = self._infonce_criterion2(proj_tf_feature, proj_feature_tf,
                                                    pos_weight=sim_coef)
        self.meters[f"{feature_name}_global/origin_mi"].add(unregularized_loss.item())
        self.meters[f"{feature_name}_global/weight_mi"].add(regularized_loss.item())

        return unregularized_loss + 0.01 * regularized_loss

    def _dense_infonce_for_encoder(self, *, feature_name, proj_tf_feature, proj_feature_tf, partition_group,
                                   label_group):
        unregularized_loss = super()._dense_infonce_for_encoder(feature_name=feature_name,
                                                                proj_tf_feature=proj_tf_feature,
                                                                proj_feature_tf=proj_feature_tf,
                                                                partition_group=partition_group,
                                                                label_group=label_group)

        # repeat a bit
        assert "Conv" in feature_name, feature_name

        proj_tf_feature, proj_feature_tf = self._reshape_dense_feature(proj_tf_feature, proj_feature_tf)

        b, hw, c = proj_feature_tf.shape

        if not (is_normalized(proj_feature_tf, dim=2) and is_normalized(proj_feature_tf, dim=2)):
            proj_feature_tf = Normalize(dim=2)(proj_feature_tf)
            proj_tf_feature = Normalize(dim=2)(proj_tf_feature)
        proj_feature_tf, proj_tf_feature = proj_feature_tf.reshape(-1, c), proj_tf_feature.reshape(-1, c)
        sim_coef = self.generate_similarity_masks(proj_feature_tf, proj_tf_feature)
        regularized_loss = self._infonce_criterion2(proj_tf_feature, proj_feature_tf,
                                                    pos_weight=sim_coef)
        self.meters[f"{feature_name}_dense/origin_mi"].add(unregularized_loss.item())
        self.meters[f"{feature_name}_dense/weight_mi"].add(regularized_loss.item())
        return unregularized_loss + 0.01 * regularized_loss

    @torch.no_grad()
    def generate_similarity_masks(self, norm_feature1: Tensor, norm_feature2: Tensor):
        sim_matrix = norm_feature1.mm(norm_feature2.transpose(1, 0))  # ranging from -1 to 1
        sim_matrix -= -1
        sim_matrix /= 2
        return sim_matrix
