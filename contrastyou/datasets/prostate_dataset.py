import os
import re
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from deepclustering2.dataset.segmentation import ProstateDataset as _ProstateDataset, \
    ProstateSemiInterface as _ProstateSemiInterface
from torch import Tensor

from contrastyou.augment.sequential_wrapper import SequentialWrapper
from contrastyou.datasets._seg_datset import ContrastDataset


class ProstateDataset(ContrastDataset, _ProstateDataset):

    def __init__(self, root_dir: str, mode: str, transforms: SequentialWrapper = SequentialWrapper(),
                 verbose=True, *args, **kwargs) -> None:
        super().__init__(root_dir, mode, ["img", "gt"], transforms, verbose, preload=True)
        self._prostate_info = np.load(os.path.join(self._root_dir, "prostate_info.npy"), allow_pickle=True).item()
        assert isinstance(self._prostate_info, dict) and len(self._prostate_info) == 50
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
        max_len_given_group = self._prostate_info[self._get_group_name(filename)]
        cutting_point = max_len_given_group // 8
        cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        return str(cur_index // (cutting_point + 1))


    def show_paritions(self) -> List[Union[str, int]]:
        return [self._get_partition(f) for f in list(self._filenames.values())[0]]

    def show_groups(self) -> List[Union[str, int]]:
        return [self._get_group(f) for f in list(self._filenames.values())[0]]


class ProstateSemiInterface(_ProstateSemiInterface):

    def __init__(self, root_dir, labeled_data_ratio: float = 0.2, unlabeled_data_ratio: float = 0.8,
                 seed: int = 0, verbose: bool = True) -> None:
        super().__init__(root_dir, labeled_data_ratio, unlabeled_data_ratio, seed, verbose)
        self.DataClass = ProstateDataset
