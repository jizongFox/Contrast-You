from PIL import Image
from contrastyou.augment.sequential_wrapper import SequentialWrapperTwice, SequentialWrapper
from deepclustering2.augment import pil_augment
from torchvision import transforms


class ACDCStrongTransforms:
    pretrain = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.RandomRotation(45),
            pil_augment.RandomVerticalFlip(),
            pil_augment.RandomHorizontalFlip(),
            pil_augment.RandomCrop(224),

        ]),
        img_transform=pil_augment.Compose([
            transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5]),
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )
    label = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(30),
        ]),
        img_transform=pil_augment.Compose([
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
    )
    val = SequentialWrapper(
        comm_transform=pil_augment.CenterCrop(224)
    )

    trainval = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),

        ]),
        img_transform=pil_augment.Compose([
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )


class ProstateStrongTransforms:
    pretrain = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.Resize(224, Image.NEAREST),
            pil_augment.RandomRotation(10),
            pil_augment.RandomVerticalFlip(),
            pil_augment.RandomHorizontalFlip(),
            pil_augment.RandomCrop(224, padding=20, ),
        ]),
        img_transform=pil_augment.Compose([
            transforms.ColorJitter(brightness=[0.9, 1.1], contrast=[0.9, 1.1], saturation=[0.9, 1.1]),
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )
    label = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.Resize(224, Image.NEAREST),
            pil_augment.RandomCrop(224),
        ]),
        img_transform=pil_augment.Compose([
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
    )
    val = SequentialWrapper(

        comm_transform=pil_augment.Resize(224, Image.NEAREST),
    )
    trainval = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),

        ]),
        img_transform=pil_augment.Compose([
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )


class SpleenStrongTransforms:
    pretrain = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.Resize((256, 256), Image.NEAREST),
            pil_augment.RandomRotation(10),
            pil_augment.RandomVerticalFlip(),
            pil_augment.RandomHorizontalFlip(),
            pil_augment.RandomCrop(256,padding=20),

        ]),
        img_transform=pil_augment.Compose([
            transforms.ColorJitter(brightness=[0.9, 1.1], contrast=[0.9, 1.1], saturation=[0.9, 1.1]),
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )
    label = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.Resize((256, 256), Image.NEAREST),
            pil_augment.RandomCrop(256,padding=20),
            pil_augment.RandomRotation(10),
        ]),
        img_transform=pil_augment.Compose([
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
    )
    val = SequentialWrapper(
        comm_transform=pil_augment.Resize((256, 256), Image.NEAREST),
    )


class MMWHSStrongTransforms:
    pretrain = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.RandomRotation(45),
            pil_augment.RandomVerticalFlip(),
            pil_augment.RandomHorizontalFlip(),
            pil_augment.RandomCrop(224),

        ]),
        img_transform=pil_augment.Compose([
            transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5]),
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )
    label = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(30),
        ]),
        img_transform=pil_augment.Compose([
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
    )
    val = SequentialWrapper(
        comm_transform=pil_augment.CenterCrop(224)
    )

    trainval = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),

        ]),
        img_transform=pil_augment.Compose([
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )
