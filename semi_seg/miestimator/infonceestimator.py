from contrastyou.arch import UNet
from contrastyou.projectors.heads import ProjectionHead, LocalProjectionHead

from ._utils import encoder_names
from .base import _SingleEstimator


# todo: unclear to see what would happen.
class InfoNCEEtimator(_SingleEstimator):
    """IICEestimator is the estimator for one single layer for the Unet"""
    __projector_initialized = False
    __criterion_initialized = False

    def init_projector(self, *,
                       layer_name: str,
                       head_type: str = "linear",
                       normalize: bool = False):
        super().__init__()
        self._layer_name = layer_name
        self._head_type = head_type
        self._normalize = normalize

        input_dim = UNet.dimension_dict[layer_name]

        CLUSTERHEAD = ProjectionHead if self._layer_name in encoder_names else LocalProjectionHead

        self._projector = CLUSTERHEAD(input_dim=input_dim, num_clusters=num_cluster,
                                      num_subheads=num_subhead, head_type=head_type, T=temperature,
                                      normalize=normalize)

        self.__projector_initialized = True

    def init_criterion(self, *, padding: int, patch_size: int):
        if self._layer_name in encoder_names:
            self._criterion = IIDLoss()
        else:
            self._criterion = IIDSegmentationSmallPathLoss(padding=padding, patch_size=patch_size)

        self.__criterion_initialized = True

    def forward(self, feat1, feat2):
        if not self.__criterion_initialized and self.__projector_initialized:
            raise RuntimeError("initialize projector and criterion first")

        return loss
