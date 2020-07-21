import random

from deepclustering2.type import to_device, torch
from deepclustering2.utils import assert_list
from torch import Tensor


def preprocess_input_with_twice_transformation(data, device, non_blocking=True):
    [(image, target), (image_tf, target_tf)], filename, partition_list, group_list = \
        to_device(data[0], device, non_blocking), data[1], data[2], data[3]
    return (image, target), (image_tf, target_tf), filename, partition_list, group_list


def preprocess_input_with_single_transformation(data, device, non_blocking=True):
    return data[0][0].to(device, non_blocking=non_blocking), data[0][1].to(device, non_blocking=non_blocking), data[1], \
           data[2], data[3]


def unfold_position(features: torch.Tensor, partition_num=(4, 4), ):
    b, c, h, w = features.shape
    block_h = h // partition_num[0]
    block_w = w // partition_num[1]
    h_index = torch.arange(0, h - block_h + 1, block_h)
    w_index = torch.arange(0, w - block_w + 1, block_w)
    result = []
    result_flag = []
    for h in h_index:
        for w in w_index:
            result.append(features[:, :, h:h + block_h, w:w + block_w])
            for _ in range(b):
                result_flag.append((int(h), int(w)))
    return torch.cat(result, dim=0), result_flag


class TensorRandomFlip:
    def __init__(self, axis=None) -> None:
        if isinstance(axis, int):
            self._axis = [axis]
        elif isinstance(axis, (list, tuple)):
            assert_list(lambda x: isinstance(x, int), axis), axis
            self._axis = axis
        elif axis is None:
            self._axis = axis
        else:
            raise ValueError(str(axis))

    def __call__(self, tensor: Tensor):
        tensor = tensor.clone()
        if self._axis is not None:
            for _one_axis in self._axis:
                if random.random() < 0.5:
                    tensor = tensor.flip(_one_axis)
            return tensor
        else:
            return tensor

    def __repr__(self):
        string = f"{self.__class__.__name__}"
        axis = "" if not self._axis else f" with axis={self._axis}."

        return string + axis


if __name__ == '__main__':
    features = torch.randn(10, 3, 256, 256, requires_grad=True)

    a = unfold_position(features, partition_num=(3, 3))
    print()
