from functools import lru_cache

from loguru import logger
from torch import nn

from .nn import _ProjectorHeadBase, Flatten, Normalize, Identical, SoftmaxWithT


# head for contrastive projection
class ProjectionHead(_ProjectorHeadBase):

    def __init__(self, *, input_dim: int, interm_dim=256, output_dim: int, head_type: str, normalize: bool,
                 pool_name="adaptive_avg", spatial_size=(1, 1)):
        assert pool_name in ("adaptive_avg", "adaptive_max")
        super().__init__(input_dim=input_dim, output_dim=output_dim, head_type=head_type, normalize=normalize,
                         pool_name=pool_name, spatial_size=spatial_size)
        if head_type == "mlp":
            self._header = nn.Sequential(
                self._pooling_module,
                Flatten(),
                nn.Linear(input_dim, interm_dim),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(interm_dim, output_dim),
                Normalize() if self._normalize else Identical()
            )
        else:
            self._header = nn.Sequential(
                self._pooling_module,
                Flatten(),
                nn.Linear(input_dim, output_dim),
                Normalize() if self._normalize else Identical()
            )
        message = self._record_message()
        logger.debug(message)

    def forward(self, features):
        return self._header(features)


# head for contrastive pixel-wise projection
class DenseProjectionHead(_ProjectorHeadBase):

    def __init__(self, *, input_dim: int, interm_dim=128, output_dim: int, head_type: str, normalize: bool,
                 pool_name="adaptive_avg", spatial_size=(16, 16)):

        super().__init__(input_dim=input_dim, output_dim=output_dim, head_type=head_type, normalize=normalize,
                         pool_name=pool_name, spatial_size=spatial_size)
        if head_type == "mlp":
            self._projector = nn.Sequential(
                nn.Conv2d(input_dim, interm_dim, 1, 1, 0),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(interm_dim, output_dim, 1, 1, 0),
            )
        else:
            self._projector = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, 1, 1, 0),
            )
        message = self._record_message()
        logger.debug(message)

    def forward(self, features):
        out = self._projector(features)
        # change resolution here
        out = self._pooling_module(out)
        if self._normalize:
            return self._normalize_func(out)
        return out

    @property
    @lru_cache()
    def _normalize_func(self):
        return Normalize()


# head for IIC clustering
class ClusterHead(_ProjectorHeadBase):

    def __init__(self, *, input_dim: int, num_clusters=5, num_subheads=10, head_type="linear", T=1, normalize=False):
        super().__init__(input_dim=input_dim, output_dim=num_clusters, head_type=head_type, normalize=normalize,
                         pool_name="none", spatial_size=(1, 1))
        self._num_clusters = num_clusters
        self._num_subheads = num_subheads
        self._T = T

        def init_sub_header(htype):
            if htype == "linear":
                return nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Linear(self._input_dim, self._num_clusters),
                    Normalize() if self._normalize else Identical(),
                    SoftmaxWithT(1, T=self._T)
                )
            else:
                return nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Linear(self._input_dim, 128),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Linear(128, num_clusters),
                    Normalize() if self._normalize else Identical(),
                    SoftmaxWithT(1, T=self._T)
                )

        headers = [
            init_sub_header(head_type)
            for _ in range(self._num_subheads)
        ]

        self._headers = nn.ModuleList(headers)
        message = self._record_message()
        logger.debug(message)

    def forward(self, features):
        return [x(features) for x in self._headers]


# head for IIC segmentation clustering
class DenseClusterHead(_ProjectorHeadBase):
    """
    this classification head uses the loss for IIC segmentation, which consists of multiple heads
    """

    def __init__(self, *, input_dim: int, num_clusters=10, interm_dim=64, num_subheads=10, T=1, head_type: str,
                 normalize: bool, ):
        super().__init__(input_dim=input_dim, output_dim=num_clusters, head_type=head_type, normalize=normalize,
                         pool_name="none", spatial_size=(1, 1))
        self._T = T

        def init_sub_header(htype):
            if htype == "linear":
                return nn.Sequential(
                    nn.Conv2d(input_dim, num_clusters, 1, 1, 0),
                    Normalize() if self._normalize else Identical(),
                    SoftmaxWithT(1, T=self._T)
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(input_dim, interm_dim, 1, 1, 0),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Conv2d(interm_dim, num_clusters, 1, 1, 0),
                    Normalize() if self._normalize else Identical(),
                    SoftmaxWithT(1, T=self._T)
                )

        headers = [init_sub_header(head_type) for _ in range(num_subheads)]
        self._headers = nn.ModuleList(headers)
        message = self._record_message()
        logger.debug(message)

    def forward(self, features):
        return [x(features) for x in self._headers]
