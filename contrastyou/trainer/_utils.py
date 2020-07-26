from torch import nn, Tensor


class Flatten(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, features):
        b, c, *_ = features.shape
        return features.view(b, -1)


class SoftmaxWithT(nn.Softmax):

    def __init__(self, dim, T:float=0.1) -> None:
        super().__init__(dim)
        self._T = T

    def forward(self, input: Tensor) -> Tensor:
        input /= self._T
        return super().forward(input)
