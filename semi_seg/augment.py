from torchvision import transforms

from contrastyou.augment import pil_augment, SequentialWrapperTwice, SequentialWrapper


class ACDCStrongTransforms:
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


class ProstateStrongTransforms:
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


class SpleenStrongTransforms:
    pretrain = SequentialWrapperTwice(
        com_transform=transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(320, padding=20),
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
            transforms.RandomRotation(10),
            transforms.RandomCrop(320, padding=20),
        ]),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel()
        ]),
    )
    val = SequentialWrapper(
        com_transform=transforms.CenterCrop(320)
    )
    trainval = SequentialWrapperTwice(
        com_transform=transforms.Compose([
            transforms.RandomCrop(320),
        ]),
        image_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        target_transform=transforms.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )


class MMWHSStrongTransforms:
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


class HippocampusStrongTransforms:
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
