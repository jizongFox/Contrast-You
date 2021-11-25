import typing as t

from torchvision import transforms

from contrastyou.augment import pil_augment, SequentialWrapperTwice, SequentialWrapper

__all__ = ["augment_zoo"]


class _Transform(t.Protocol):
    pretrain: SequentialWrapperTwice
    label: SequentialWrapperTwice
    val: SequentialWrapper
    trainval: SequentialWrapperTwice


class ACDCStrongTransforms(_Transform):
    pretrain = SequentialWrapperTwice(
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
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )
    label = SequentialWrapperTwice(
        com_transform=transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomRotation(30),
        ]),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel()
        ]),
    )
    val = SequentialWrapper(
        com_transform=transforms.CenterCrop(224)
    )

    trainval = SequentialWrapperTwice(
        com_transform=transforms.Compose([
            transforms.RandomCrop(224),

        ]),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )


class ACDCLVStrongTransforms(_Transform):
    pretrain = SequentialWrapperTwice(
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
            pil_augment.ToLabel({0: 0, 1: 0, 2: 0, 3: 1})
        ]),
        total_freedom=True
    )
    label = SequentialWrapperTwice(
        com_transform=transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomRotation(30),
        ]),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel({0: 0, 1: 0, 2: 0, 3: 1})
        ]),
    )
    val = SequentialWrapper(
        com_transform=transforms.CenterCrop(224),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel({0: 0, 1: 0, 2: 0, 3: 1})
        ]),
    )

    trainval = SequentialWrapperTwice(
        com_transform=transforms.Compose([
            transforms.RandomCrop(224),

        ]),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel({0: 0, 1: 0, 2: 0, 3: 1})
        ]),
        total_freedom=True
    )


class ACDCRVStrongTransforms(_Transform):
    pretrain = SequentialWrapperTwice(
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
            pil_augment.ToLabel({0: 0, 1: 1, 2: 0, 3: 0})
        ]),
        total_freedom=True
    )
    label = SequentialWrapperTwice(
        com_transform=transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomRotation(30),
        ]),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel({0: 0, 1: 1, 2: 0, 3: 0})
        ]),
    )
    val = SequentialWrapper(
        com_transform=transforms.CenterCrop(224),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel({0: 0, 1: 1, 2: 0, 3: 0})
        ]),
    )

    trainval = SequentialWrapperTwice(
        com_transform=transforms.Compose([
            transforms.RandomCrop(224),

        ]),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel({0: 0, 1: 1, 2: 0, 3: 0})
        ]),
        total_freedom=True
    )


class ProstateStrongTransforms(_Transform):
    pretrain = SequentialWrapperTwice(
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
    label = SequentialWrapperTwice(
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
    val = SequentialWrapper(

        com_transform=transforms.Resize(224, ),
    )
    trainval = SequentialWrapperTwice(
        com_transform=transforms.Compose([
            transforms.RandomCrop(224),

        ]),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )


class SpleenStrongTransforms(_Transform):
    pretrain = SequentialWrapperTwice(
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
    label = SequentialWrapperTwice(
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
    val = SequentialWrapper(
        com_transform=transforms.Compose([transforms.Resize(320), transforms.CenterCrop(256)])
    )
    trainval = SequentialWrapperTwice(
        com_transform=transforms.Compose([
            transforms.Resize(320),
            transforms.RandomCrop(256),
        ]),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )


class MMWHSStrongTransforms(_Transform):
    pretrain = SequentialWrapperTwice(
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
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )
    label = SequentialWrapperTwice(
        com_transform=transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomRotation(30),
        ]),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel()
        ]),
    )
    val = SequentialWrapper(
        com_transform=transforms.CenterCrop(224)
    )

    trainval = SequentialWrapperTwice(
        com_transform=transforms.Compose([
            transforms.RandomCrop(224),

        ]),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )


class HippocampusStrongTransforms(_Transform):
    pretrain = SequentialWrapperTwice(
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
    label = SequentialWrapperTwice(
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
    val = SequentialWrapper(
        com_transform=transforms.Resize((64, 64), ),
    )
    trainval = SequentialWrapperTwice(
        com_transform=transforms.Compose([
            transforms.RandomCrop(64),

        ]),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )


augment_zoo: t.Dict[str, _Transform] = {
    "acdc": ACDCStrongTransforms, "acdc_lv": ACDCLVStrongTransforms, "acdc_rv": ACDCRVStrongTransforms,
    "spleen": SpleenStrongTransforms, "prostate": ProstateStrongTransforms, "mmwhsct": ACDCStrongTransforms,
    "mmwhsmr": ACDCStrongTransforms, "prostate_md": ProstateStrongTransforms, "hippocampus": HippocampusStrongTransforms
}
