import random
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, List, Tuple, TypeVar, Iterable

from PIL import Image
from torch import Tensor
from torchvision.transforms import Compose

from . import pil_augment

__all__ = ["SequentialWrapper", "SequentialWrapperTwice"]

from ..helper import fix_all_seed_for_transforms

_pil2pil_transform_type = Callable[[Image.Image], Image.Image]
_pil2tensor_transform_type = Callable[[Image.Image], Tensor]
_pil_list = List[Image.Image]

T = TypeVar("T")


def get_transform(transform) -> Iterable[Callable[[T], T]]:
    if isinstance(transform, Compose):
        for x in transform.transforms:
            yield from get_transform(x)
    else:
        yield transform


@contextmanager
def switch_interpolation(transforms: Callable[[T], T], *, interp: str):
    assert interp in ("bilinear", "nearest"), interp
    interpolation = {"bilinear": Image.BILINEAR, "nearest": Image.NEAREST}[interp]

    previous_inters = OrderedDict()
    transforms = get_transform(transforms)
    for id, t in enumerate(transforms):
        if hasattr(t, "interpolation"):
            previous_inters[id] = t.interpolation
            t.interpolation = interpolation
    yield
    transforms = get_transform(transforms)
    for id, t in enumerate(transforms):
        if hasattr(t, "interpolation"):
            t.interpolation = previous_inters[id]


def random_int() -> int:
    return random.randint(0, int(1e5))


def transform_(image, transform, seed):
    with fix_all_seed_for_transforms(seed):
        return transform(image)


class SequentialWrapper:

    def __init__(self, com_transform: _pil2pil_transform_type = None,
                 image_transform: _pil2tensor_transform_type = pil_augment.ToTensor(),
                 target_transform: _pil2tensor_transform_type = pil_augment.ToLabel()) -> None:
        """
        image -> comm_transform -> img_transform -> Tensor
        target -> comm_transform -> target_transform -> Tensor
        :param com_transform: common geo-transformation
        :param image_transform: transformation only applied for images
        :param target_transform: transformation only applied for targets
        """
        self._com_transform = com_transform
        self._image_transform = image_transform
        self._target_transform = target_transform

    def __call__(self, images: _pil_list, targets: _pil_list = None, com_seed: int = None,
                 img_seed: int = None, target_seed: int = None) -> Tuple[List[Tensor], List[Tensor]]:
        com_seed = com_seed or random_int()
        img_seed = img_seed or random_int()
        target_seed = target_seed or random_int()

        image_list_after_transform, target_list_after_transform = images, targets or []

        if self._com_transform:
            # comm is the optional
            with switch_interpolation(self._com_transform, interp="bilinear"):
                image_list_after_transform = [transform_(image, self._com_transform, com_seed) for image in
                                              image_list_after_transform]
            if targets is not None:
                with switch_interpolation(self._com_transform, interp="nearest"):
                    target_list_after_transform = [transform_(target, self._com_transform, com_seed) for target in
                                                   target_list_after_transform]

        image_list_after_transform = [transform_(image, self._image_transform, img_seed) for image in
                                      image_list_after_transform]
        if targets is not None:
            target_list_after_transform = [transform_(target, self._target_transform, target_seed) for target in
                                           target_list_after_transform]

        return image_list_after_transform, target_list_after_transform

    def __repr__(self):
        return (
            f"comm_transform:{self._com_transform}\n"
            f"img_transform:{self._image_transform}.\n"
            f"target_transform: {self._target_transform}"
        )


class SequentialWrapperTwice(SequentialWrapper):

    def __init__(self, com_transform: _pil2pil_transform_type = None,
                 image_transform: _pil2tensor_transform_type = pil_augment.ToTensor(),
                 target_transform: _pil2tensor_transform_type = pil_augment.ToLabel(),
                 total_freedom=True) -> None:
        """
        :param total_freedom: if True, the two-time generated images are using different seeds for all aspect,
                              otherwise, the images are used different random seed only for img_seed
        """
        super().__init__(com_transform, image_transform, target_transform)
        self._total_freedom = total_freedom

    def __call__(self, image_list: _pil_list, target_list: _pil_list = None, seed: int = None, **kwargs) -> \
        Tuple[List[Tensor], List[Tensor]]:
        seed = seed or random_int()

        with fix_all_seed_for_transforms(seed):
            comm_seed1, comm_seed2 = random_int(), random_int()
            img_seed1, img_seed2 = random_int(), random_int()
            target_seed1, target_seed2 = random_int(), random_int()

            if self._total_freedom:
                image_list1, target_list1 = super(SequentialWrapperTwice, self).__call__(image_list, target_list,
                                                                                         comm_seed1, img_seed1,
                                                                                         target_seed1)
                image_list2, target_list2 = super(SequentialWrapperTwice, self).__call__(image_list, target_list,
                                                                                         comm_seed2, img_seed2,
                                                                                         target_seed2)
                return [*image_list1, *image_list2], [*target_list1, *target_list2]

            image_list1, target_list1 = super(SequentialWrapperTwice, self).__call__(image_list, target_list,
                                                                                     comm_seed1, img_seed1,
                                                                                     target_seed1)
            image_list2, target_list2 = super(SequentialWrapperTwice, self).__call__(image_list, target_list,
                                                                                     comm_seed1, img_seed2,
                                                                                     target_seed1)
            return [*image_list1, *image_list2], [*target_list1, *target_list2]
