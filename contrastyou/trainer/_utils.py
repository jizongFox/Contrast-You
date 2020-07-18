from torch import nn

class Flatten(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, features):
        b, c, *_ = features.shape
        return features.view(b, -1)