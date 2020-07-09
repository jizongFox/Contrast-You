from typing import Callable, Union, List, Tuple

from deepclustering.augment import pil_augment, SequentialWrapper


class SequentialWrapperTwice(SequentialWrapper):

    def __init__(self, img_transform: Callable = None, target_transform: Callable = None,
                 if_is_target: Union[List[bool], Tuple[bool, ...]] = []) -> None:
        super().__init__(img_transform, target_transform, if_is_target)

    def __call__(
        self, *imgs, random_seed=None
    ):
        return [
            super(SequentialWrapperTwice, self).__call__(*imgs, random_seed=random_seed),
            super(SequentialWrapperTwice, self).__call__(*imgs, random_seed=random_seed),
        ]


class ACDC_transforms:
    train = SequentialWrapperTwice(
        pil_augment.Compose([
            pil_augment.CenterCrop(224),
            pil_augment.RandomRotation(5),
            pil_augment.ToTensor()
        ]),
        pil_augment.Compose([
            pil_augment.CenterCrop(224),
            pil_augment.RandomRotation(5),
            pil_augment.ToTensor()
        ]),
        if_is_target=[False, True]

    )
    val = SequentialWrapperTwice(
        pil_augment.Compose([
            pil_augment.CenterCrop(224),
            pil_augment.ToTensor()
        ]),
        pil_augment.Compose([
            pil_augment.CenterCrop(224),
            pil_augment.ToTensor()
        ]),
        if_is_target=[False, True]

    )
