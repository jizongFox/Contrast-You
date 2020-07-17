import os
import re
from typing import List, Tuple, Union

import numpy as np
from torch import Tensor

from contrastyou.dataloader._seg_datset import ContrastDataset
from deepclustering2.augment import SequentialWrapper
from deepclustering2.dataset import ACDCDataset as _ACDCDataset, ACDCSemiInterface as _ACDCSemiInterface


class ACDCDataset(ContrastDataset, _ACDCDataset):
    download_link = "https://drive.google.com/uc?id=1SMAS6R46BOafLKE9T8MDSVGAiavXPV-E"
    zip_name = "ACDC_contrast.zip"
    folder_name = "ACDC_contrast"

    def __init__(self, root_dir: str, mode: str, transforms: SequentialWrapper = None,
                 verbose=True, *args, **kwargs) -> None:
        super().__init__(root_dir, mode, ["img", "gt"], transforms, verbose)
        self._acdc_info = np.load(os.path.join(self._root_dir, "acdc_info.npy"), allow_pickle=True).item()
        assert isinstance(self._acdc_info, dict) and len(self._acdc_info) == 200

    def __getitem__(self, index) -> Tuple[List[Tensor], str, str, str]:
        data, filename = super().__getitem__(index)
        partition = self._get_partition(filename)
        group = self._get_group(filename)
        return data, filename, partition, group

    def _get_group(self, filename) -> Union[str, int]:
        return self._get_group_name(filename)

    def _get_partition(self, filename) -> Union[str, int]:
        # set partition
        max_len_given_group = self._acdc_info[self._get_group_name(filename)]
        cutting_point = max_len_given_group // 3
        cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        if cur_index <= cutting_point - 1:
            return 0
        if cur_index <= 2 * cutting_point:
            return 1
        return 2

    def show_paritions(self) -> List[Union[str, int]]:
        return [self._get_partition(f) for f in list(self._filenames.values())[0]]

    def show_groups(self) -> List[Union[str, int]]:
        return [self._get_group(f) for f in list(self._filenames.values())[0]]


class ACDCSemiInterface(_ACDCSemiInterface):

    def __init__(self, root_dir, labeled_data_ratio: float = 0.2, unlabeled_data_ratio: float = 0.8,
                 seed: int = 0, verbose: bool = True) -> None:
        super().__init__(root_dir, labeled_data_ratio, unlabeled_data_ratio, seed, verbose)
        self.DataClass = ACDCDataset
