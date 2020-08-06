from typing import Union

from torch.utils.data import Subset
from torchvision import datasets


class CIFAR10(datasets.CIFAR10):
    @classmethod
    def create_semi_dataset(cls, root, train=True, transform=None, target_transform=None, download=False, selected_index_list=None) -> Union[
        datasets.CIFAR10, Subset]:
        dataset = cls(root, train, transform, target_transform, download)
        if selected_index_list is None:
            return dataset
        return Subset(dataset, selected_index_list)
