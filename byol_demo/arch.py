from torch import nn
from torchvision import models


class Resnet(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super().__init__()
        resnet_50 = models.resnet50()
        self._feature_extractor = nn.Sequential(*list(resnet_50.children())[:-1])
        self._classhead = nn.Linear(2048, num_classes)
        self._prediction = nn.Linear(2048, 2048)

    def forward(self, image, return_features=False, return_classes=False, return_predictions=False):
        features = self._feature_extractor(image)
        if return_features:
            return features
        if return_classes:
            return self._classhead(features)
        if return_predictions:
            return self._prediction(features)
        raise TypeError


if __name__ == '__main__':
    Resnet()
