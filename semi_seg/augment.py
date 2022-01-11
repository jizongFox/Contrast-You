import typing as t
from functools import partial

import rising.transforms as rt
from torch import Tensor
from torchvision import transforms

from contrastyou.augment import pil_augment, SequentialWrapperTwice, SequentialWrapper
from contrastyou.utils import fix_all_seed_for_transforms

__all__ = ["augment_zoo", "RisingWrapper"]


class _Transform(t.Protocol):
    @property
    def pretrain(self) -> SequentialWrapperTwice:
        raise NotImplemented

    @property
    def label(self) -> SequentialWrapperTwice:
        raise NotImplemented

    @property
    def val(self) -> SequentialWrapper:
        raise NotImplemented

    @property
    def trainval(self) -> SequentialWrapperTwice:
        raise NotImplemented


class ACDCTransforms2(_Transform):

    def __init__(self, mapping: t.Dict[int, int] = None) -> None:
        super().__init__()
        self.mapping = mapping

    @property
    def pretrain(self):
        return SequentialWrapperTwice(
            com_transform=transforms.Compose([
                transforms.RandomRotation(45),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224),
            ]),
            image_transform=transforms.Compose([
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5]),
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                pil_augment.ToLabel(mapping=self.mapping)
            ]),
            total_freedom=True
        )

    @property
    def label(self):
        return SequentialWrapperTwice(
            com_transform=transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomRotation(30),
            ]),
            image_transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                pil_augment.ToLabel(mapping=self.mapping)
            ]),
        )

    @property
    def val(self):
        return SequentialWrapper(
            com_transform=transforms.CenterCrop(224),
            image_transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                pil_augment.ToLabel(mapping=self.mapping)
            ]),
        )

    @property
    def trainval(self):
        return SequentialWrapperTwice(
            com_transform=transforms.Compose([
                transforms.CenterCrop(224),

            ]),
            image_transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                pil_augment.ToLabel(mapping=self.mapping)
            ]),
            total_freedom=True
        )


class ProstateTransforms(_Transform):
    @property
    def pretrain(self):
        return SequentialWrapperTwice(
            com_transform=transforms.Compose([
                transforms.Resize(224, ),
                transforms.RandomRotation(10),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=20, ),
            ]),
            image_transform=transforms.Compose([
                transforms.ColorJitter(brightness=[0.9, 1.1], contrast=[0.9, 1.1], saturation=[0.9, 1.1]),
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                pil_augment.ToLabel()
            ]),
            total_freedom=True
        )

    @property
    def label(self):
        return SequentialWrapperTwice(
            com_transform=transforms.Compose([
                transforms.Resize(224, ),
                transforms.RandomCrop(224),
            ]),
            image_transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                pil_augment.ToLabel()
            ]),
        )

    @property
    def val(self):
        return SequentialWrapper(
            com_transform=transforms.Resize(224, ),
        )

    @property
    def trainval(self):
        return SequentialWrapperTwice(
            com_transform=transforms.Compose([
                transforms.CenterCrop(224),

            ]),
            image_transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                pil_augment.ToLabel()
            ]),
            total_freedom=True
        )


class SpleenTransforms(_Transform):
    @property
    def pretrain(self):
        return SequentialWrapperTwice(
            com_transform=transforms.Compose([
                transforms.Resize(320),
                transforms.RandomRotation(30),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(256, padding=20),
            ]),
            image_transform=transforms.Compose([
                transforms.ColorJitter(brightness=[0.9, 1.1], contrast=[0.9, 1.1], saturation=[0.9, 1.1]),
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                pil_augment.ToLabel()
            ]),
            total_freedom=True
        )

    @property
    def label(self):
        return SequentialWrapperTwice(
            com_transform=transforms.Compose([
                transforms.Resize(320),
                transforms.RandomRotation(10),
                transforms.RandomCrop(256, padding=20),
            ]),
            image_transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                pil_augment.ToLabel()
            ]),
        )

    @property
    def val(self):
        return SequentialWrapper(
            com_transform=transforms.Compose([transforms.Resize(320), transforms.CenterCrop(256)])
        )

    @property
    def trainval(self) -> SequentialWrapperTwice:
        return SequentialWrapperTwice(
            com_transform=transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(256),
            ]),
            image_transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                pil_augment.ToLabel()
            ]),
            total_freedom=True
        )


class HippocampusTransforms(_Transform):

    @property
    def pretrain(self) -> SequentialWrapperTwice:
        return SequentialWrapperTwice(
            com_transform=transforms.Compose([
                transforms.Resize((64, 64), ),
                transforms.RandomRotation(10),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=20),

            ]),
            image_transform=transforms.Compose([
                transforms.ColorJitter(brightness=[0.9, 1.1], contrast=[0.9, 1.1], saturation=[0.9, 1.1]),
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                pil_augment.ToLabel()
            ]),
            total_freedom=True
        )

    @property
    def label(self) -> SequentialWrapperTwice:
        return SequentialWrapperTwice(
            com_transform=transforms.Compose([
                transforms.Resize((64, 64), ),
                transforms.RandomCrop(64, padding=20),
                transforms.RandomRotation(10),
            ]),
            image_transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                pil_augment.ToLabel()
            ]),
        )

    @property
    def val(self) -> SequentialWrapper:
        return SequentialWrapper(
            com_transform=transforms.Resize((64, 64), ),
        )

    @property
    def trainval(self) -> SequentialWrapperTwice:
        return SequentialWrapperTwice(
            com_transform=transforms.Compose([
                transforms.CenterCrop(64),

            ]),
            image_transform=transforms.Compose([
                transforms.ToTensor()
            ]),
            target_transform=transforms.Compose([
                pil_augment.ToLabel()
            ]),
            total_freedom=True
        )


class RisingWrapper:

    def __init__(
            self,
            *,
            geometry_transform: rt._AbstractTransform = None,
            intensity_transform: rt._AbstractTransform = None
    ) -> None:
        self.geometry_transform = geometry_transform
        self.intensity_transform = intensity_transform

    def __call__(self, image: Tensor, *, mode: str, seed: int):
        assert mode in ("image", "feature"), f"`mode` must be in `image` or `feature`, given {mode}."
        if mode == "image":
            with fix_all_seed_for_transforms(seed):
                if self.intensity_transform is not None:
                    image = self.intensity_transform(data=image)["data"]
            with fix_all_seed_for_transforms(seed):
                if self.geometry_transform is not None:
                    image = self.geometry_transform(data=image)["data"]
        else:
            with fix_all_seed_for_transforms(seed):
                if self.geometry_transform is not None:
                    image = self.geometry_transform(data=image)["data"]
        return image


augment_zoo: t.Dict[str, t.Type[_Transform]] = {
    "acdc": partial(ACDCTransforms2, mapping=None),
    "acdc_lv": partial(ACDCTransforms2, mapping={0: 0, 1: 0, 2: 0, 3: 1}),
    "acdc_rv": partial(ACDCTransforms2, mapping={0: 0, 1: 1, 2: 0, 3: 0}),
    "acdc_myo": partial(ACDCTransforms2, mapping={0: 0, 1: 0, 2: 1, 3: 0}),
    "spleen": SpleenTransforms,
    "prostate": ProstateTransforms,
    "mmwhsct": partial(ACDCTransforms2, mapping=None),
    "mmwhsmr": partial(ACDCTransforms2, mapping=None),
    "prostate_md": ProstateTransforms,
    "hippocampus": HippocampusTransforms
}
