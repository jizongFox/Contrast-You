from torch import nn, Tensor
from torch.nn import functional as F


class Flatten(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, features):
        b, *_ = features.shape
        return features.view(b, -1)


class SoftmaxWithT(nn.Softmax):

    def __init__(self, dim, T: float = 1.0) -> None:
        super().__init__(dim)
        self._T = T

    def forward(self, input: Tensor) -> Tensor:
        input /= self._T
        return super().forward(input)


class Normalize(nn.Module):

    def __init__(self, dim=1) -> None:
        super().__init__()
        self._dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self._dim)


class Identical(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return input


class ProjectorHeadBase(nn.Module):
    pass
