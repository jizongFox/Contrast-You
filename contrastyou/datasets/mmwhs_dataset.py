import re
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from deepclustering2.dataset.segmentation.mmwhs_dataset import MMWHSDataset as _MMWHSDataset, \
    MMWHSSemiInterface as _MMWHSSemiInterface
from torch import Tensor

from contrastyou import DATA_PATH
from contrastyou.augment.sequential_wrapper import SequentialWrapper
from contrastyou.datasets._seg_datset import ContrastDataset


class MMWHSDataset(ContrastDataset, _MMWHSDataset):

    def __init__(self, root_dir: str, modality: str, mode: str, subfolders=["img", "gt"],
                 transforms: SequentialWrapper = None, verbose=True) -> None:
        super().__init__(root_dir, modality, mode, subfolders, transforms, verbose)
        self._transform = transforms
        self._meta_info = {"ct": np.load(str(Path(root_dir, "MMWHS", "meta_ct.npy")), allow_pickle=True).tolist(),
                           "mr": np.load(str(Path(root_dir, "MMWHS", "meta_mr.npy")), allow_pickle=True).tolist()}

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
        max_len_given_group = int(self._meta_info[self._mode.split("_")[0]][self._get_group_name(filename)])
        cutting_point = max_len_given_group // 8
        cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        return str(cur_index // (cutting_point + 1))

    def show_paritions(self) -> List[Union[str, int]]:
        return [self._get_partition(f) for f in list(self._filenames.values())[0]]

    def show_groups(self) -> List[Union[str, int]]:
        return [self._get_group(f) for f in list(self._filenames.values())[0]]


class MMWHSSemiInterface(_MMWHSSemiInterface):

    def __init__(self, root_dir=DATA_PATH, modality="ct", labeled_data_ratio: float = 0.2,
                 unlabeled_data_ratio: float = 0.8, seed: int = 0, verbose: bool = True) -> None:
        super().__init__(root_dir, modality, labeled_data_ratio, unlabeled_data_ratio, seed, verbose)
        self.DataClass = MMWHSDataset
