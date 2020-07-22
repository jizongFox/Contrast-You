import torch
from torch import nn
from torchvision import models


class Resnet(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        resnet_50 = models.resnet50()
        self._feature_extractor = nn.Sequential(*list(resnet_50.children())[:-1])
        self._classhead = nn.Linear(2048, num_classes)
        self._projection = nn.Linear(2048, 1024)
        self._prediction = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024)
        )

    def forward(self, image, return_projections=False, return_classes=False, return_predictions=False):
        features = torch.flatten(self._feature_extractor(image), 1)
        if return_classes:
            return self._classhead(features)
        if return_projections:
            return self._projection(features)
        if return_predictions:
            return self._prediction(self._projection(features))
        raise TypeError


if __name__ == '__main__':
    Resnet()
