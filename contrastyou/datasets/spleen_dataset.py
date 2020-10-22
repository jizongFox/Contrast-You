import os
import re
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from torch import Tensor

from contrastyou.augment.sequential_wrapper import SequentialWrapper
from contrastyou.datasets._seg_datset import ContrastDataset
from deepclustering2.dataset.segmentation.spleen_dataset import SpleenDataset as _SpleenDataset, \
    SpleenSemiInterface as _SpleenSemiInterface


class SpleenDataset(ContrastDataset, _SpleenDataset):

    def __init__(self, root_dir: str, mode: str, transforms: SequentialWrapper = SequentialWrapper(),
                 verbose=True, *args, **kwargs) -> None:
        super().__init__(root_dir, mode, ["img", "gt"], transforms, verbose)
        self._transform = transforms

    def __getitem__(self, index) -> Tuple[List[Tensor], str, str, str]:
        [img_png, target_png], filename_list = self._getitem_index(index)
        filename = Path(filename_list[0]).stem
        data = self._transform(imgs=[img_png], targets=[target_png], )
        partition = self._get_partition(filename)
        group = self._get_group(filename)
        return data, filename, partition, group

    def _get_group(self, filename) -> Union[str, int]:
        return str(self._get_group_name(filename))

    def _get_partition(self, filename) -> Union[str, int]:
        # set partition
        # max_len_given_group = self._acdc_info[self._get_group_name(filename)]
        # cutting_point = max_len_given_group // 3
        # cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        # if cur_index <= cutting_point - 1:
        #     return str(0)
        # if cur_index <= 2 * cutting_point:
        #     return str(1)
        # return str(2)
        return str(0)

    def show_paritions(self) -> List[Union[str, int]]:
        return [self._get_partition(f) for f in list(self._filenames.values())[0]]

    def show_groups(self) -> List[Union[str, int]]:
        return [self._get_group(f) for f in list(self._filenames.values())[0]]


class SpleenSemiInterface(_SpleenSemiInterface):

    def __init__(self, root_dir, labeled_data_ratio: float = 0.2, unlabeled_data_ratio: float = 0.8,
                 seed: int = 0, verbose: bool = True) -> None:
        super().__init__(root_dir, labeled_data_ratio, unlabeled_data_ratio, seed, verbose)
        self.DataClass = SpleenDataset
