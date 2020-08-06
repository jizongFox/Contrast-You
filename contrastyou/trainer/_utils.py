from torch import nn, Tensor


class Flatten(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, features):
        b, c, *_ = features.shape
        return features.view(b, -1)


class SoftmaxWithT(nn.Softmax):

    def __init__(self, dim, T: float = 0.1) -> None:
        super().__init__(dim)
        self._T = T

    def forward(self, input: Tensor) -> Tensor:
        input /= self._T
        return super().forward(input)


class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_clusters=5, num_subheads=10, head_type="mlp", T=1) -> None:
        super().__init__()
        assert head_type in ("linear", "mlp"), head_type
        self._input_dim = input_dim
        self._num_clusters = num_clusters
        self._num_subheads = num_subheads
        self._T = T

        def init_sub_header(head_type):
            if head_type == "linear":
                return nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Linear(self._input_dim, self._num_clusters),
                    SoftmaxWithT(1, T=self._T)
                )
            else:
                return nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Linear(self._input_dim, 512),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Linear(512, num_clusters),
                    SoftmaxWithT(1, T=self._T)
                )

        headers = [
            init_sub_header(head_type)
            for _ in range(self._num_subheads)
        ]

        self._headers = nn.ModuleList(headers)

    def forward(self, features):
        return [x(features) for x in self._headers]


class ProjectionHead(nn.Module):

    def __init__(self, input_dim, output_dim, interm_dim=256, head_type="mlp") -> None:
        super().__init__()
        assert head_type in ("mlp", "linear")
        if head_type == "mlp":
            self._header = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(input_dim, interm_dim),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(interm_dim, output_dim),
            )
        else:
            self._header = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Linear(input_dim, output_dim),
            )
    def forward(self, features):
        return self._header(features)