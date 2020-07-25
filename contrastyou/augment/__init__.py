from torchvision import transforms

from contrastyou.augment.sequential_wrapper import SequentialWrapperTwice, SequentialWrapper
from deepclustering2.augment import pil_augment


class ACDCTransforms:
    train = SequentialWrapperTwice(
        comm_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(30),
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
    val = SequentialWrapper(
        comm_transform=pil_augment.CenterCrop(224)
    )
