from functools import lru_cache

from torch import nn
from torch.nn import functional as F

from .nn import HeadBase, Flatten, Normalize, Identical, SoftmaxWithT


def _check_head_type(head_type):
    return head_type in ("mlp", "linear")


# head for contrastive projection
class ProjectionHead(HeadBase):

    def __init__(self, input_dim, output_dim, interm_dim=256, head_type="mlp", normalize=True) -> None:
        super().__init__()
        assert _check_head_type(head_type), head_type
        self._normalize = normalize
        if head_type == "mlp":
            self._header = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(input_dim, interm_dim),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(interm_dim, output_dim),
                Normalize() if self._normalize else Identical()
            )
        else:
            self._header = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(input_dim, output_dim),
                Normalize() if self._normalize else Identical()
            )

    def forward(self, features):
        return self._header(features)


# head for contrastive pixel-wise projection
class LocalProjectionHead(HeadBase):
    """
    return a fixed feature size
    """

    def __init__(self, input_dim, head_type="mlp", output_size=(4, 4), normalize=True) -> None:
        super().__init__()
        assert _check_head_type(head_type), head_type
        self._output_size = output_size
        self._normalize = normalize
        if head_type == "mlp":
            self._projector = nn.Sequential(
                nn.Conv2d(input_dim, 64, 3, 1, 1),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv2d(64, 32, 3, 1, 1),
            )
        else:
            self._projector = nn.Sequential(
                nn.Conv2d(input_dim, 64, 3, 1, 1),
            )

    def forward(self, features):
        b, c, h, w = features.shape
        out = self._projector(features)
        # fixme: Upsampling and interpolate don't pass the gradient correctly.
        out = F.adaptive_max_pool2d(out, output_size=self._output_size)
        if self._normalize:
            return self._normalize_func(out)
        return out

    @property
    @lru_cache()
    def _normalize_func(self):
        return Normalize()


# head for IIC clustering
class ClusterHead(HeadBase):
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
class LocalClusterHead(HeadBase):
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
