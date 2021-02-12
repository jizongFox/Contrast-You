from functools import lru_cache

from loguru import logger
from torch import nn

from .nn import ProjectorHeadBase, Flatten, Normalize, Identical, SoftmaxWithT


def _check_head_type(head_type):
    return head_type in ("mlp", "linear")


# head for contrastive projection
class ProjectionHead(ProjectorHeadBase):

    def __init__(self, input_dim, output_dim, interm_dim=256, head_type="mlp", normalize=True,
                 pooling_name: str = None) -> None:
        super().__init__()
        assert _check_head_type(head_type), head_type
        assert pooling_name in ("adaptive_avg", "adaptive_max"), pooling_name
        self._normalize = normalize
        pooling_module = {"adaptive_avg": nn.AdaptiveAvgPool2d((1, 1)),
                          "adaptive_max": nn.AdaptiveMaxPool2d((1, 1))}[pooling_name]
        if head_type == "mlp":
            self._header = nn.Sequential(
                pooling_module,
                Flatten(),
                nn.Linear(input_dim, interm_dim),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(interm_dim, output_dim),
                Normalize() if self._normalize else Identical()
            )
        else:
            self._header = nn.Sequential(
                pooling_module,
                Flatten(),
                nn.Linear(input_dim, output_dim),
                Normalize() if self._normalize else Identical()
            )
        logger.debug("initialize {} with pooling method of: {}", self.__class__.__name__, pooling_name)

    def forward(self, features):
        return self._header(features)


# head for contrastive pixel-wise projection
class DenseProjectionHead(ProjectorHeadBase):
    """
    return a fixed feature size
    """

    def __init__(self, input_dim, head_type="mlp", output_size=(4, 4), normalize=True,
                 pooling_name="adaptive_avg") -> None:
        """
        :param input_dim:
        :param head_type:
        :param output_size: Tuple of dimension, default (4,4)
        :param normalize: if normalize the dense dimension.
        :param pooling_name: adaptive_avg or adaptive_max or none
        """
        super().__init__()
        assert _check_head_type(head_type), head_type
        self._output_size = output_size
        self._normalize = normalize
        self._pooling_name = pooling_name

        self._pooling_module = {
            "adaptive_avg": nn.AdaptiveAvgPool2d(output_size),
            "adaptive_max": nn.AdaptiveMaxPool2d(output_size),
            None: Identical(),
            "none": Identical()
        }[pooling_name]
        if pooling_name in ("adaptive_avg", "adaptive_max"):
            logger.debug("initialize {} with pooling of {} and output_size of: {}",
                         self.__class__.__name__, self._pooling_module.__class__.__name__, output_size)
        else:
            logger.debug("initialize {} without pooling", self.__class__.__name__)

        if head_type == "mlp":
            self._projector = nn.Sequential(
                nn.Conv2d(input_dim, 64, 1, 1, 0),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(64, 32, 1, 1, 0),
            )
        else:
            self._projector = nn.Sequential(
                nn.Conv2d(input_dim, 64, 1, 1, 0),
            )

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
class ClusterHead(ProjectorHeadBase):
    def __init__(self, input_dim, num_clusters=5, num_subheads=10, head_type="linear", T=1, normalize=False) -> None:
        super().__init__()
        assert _check_head_type(head_type), head_type
        self._input_dim = input_dim
        self._num_clusters = num_clusters
        self._num_subheads = num_subheads
        self._T = T
        self._normalize = normalize

        def init_sub_header(head_type):
            if head_type == "linear":
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

    def forward(self, features):
        return [x(features) for x in self._headers]


# head for IIC segmentaiton clustering
class DenseClusterHead(ProjectorHeadBase):
    """
    this classification head uses the loss for IIC segmentation, which consists of multiple heads
    """

    def __init__(self, input_dim, head_type="linear", num_clusters=10, num_subheads=10, T=1, interm_dim=64,
                 normalize=False) -> None:
        super().__init__()
        assert _check_head_type(head_type), head_type
        self._T = T
        self._normalize = normalize

        def init_sub_header(head_type):
            if head_type == "linear":
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

    def forward(self, features):
        return [x(features) for x in self._headers]
