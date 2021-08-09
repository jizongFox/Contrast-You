import os
import re
from pathlib import Path
from typing import Tuple, List, Callable, Dict

import numpy as np
from torch import Tensor

from contrastyou.augment import SequentialWrapper
from contrastyou.data import ACDCDataset as _acdc, ProstateDataset as _prostate, mmWHSCTDataset as _mmct, \
    mmWHSMRDataset as _mmmr, ProstateMDDataset as _prostate_md, SpleenDataset as _spleen, \
    HippocampusDataset as _Hippocampus
from .rearr import ContrastDataset


class ACDCDataset(ContrastDataset, _acdc):
    download_link = "https://drive.google.com/uc?id=1SMAS6R46BOafLKE9T8MDSVGAiavXPV-E"
    zip_name = "ACDC_contrast.zip"
    folder_name = "ACDC_contrast"
    partition_num = 3

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        super().__init__(root_dir=root_dir, mode=mode, transforms=transforms)
        self._acdc_info = np.load(os.path.join(self._root_dir, "acdc_info.npy"),
                                  allow_pickle=True).item()
        assert isinstance(self._acdc_info, dict) and len(self._acdc_info) == 200

    def __getitem__(self, index) -> Tuple[List[Tensor], str, Tuple[str, str]]:
        images, filename = super().__getitem__(index)
        partition = self._get_partition(filename)
        scan_num = self._get_scan_name(filename)
        return images, filename, (partition, scan_num)

    def _get_partition(self, stem) -> str:
        # set partition
        max_len_given_group = self._acdc_info[self._get_scan_name(stem)]  # noqa
        cutting_point = max_len_given_group // self.partition_num
        cur_index = int(re.compile(r"\d+").findall(stem)[-1])
        if cur_index <= cutting_point - 1:
            return str(0)
        if cur_index <= 2 * cutting_point:
            return str(1)
        return str(2)


class ProstateDataset(ContrastDataset, _prostate):
    partition_num = 8

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        super().__init__(root_dir=root_dir, mode=mode, transforms=transforms)
        self._prostate_info: Dict[str, int] \
            = np.load(os.path.join(self._root_dir, "prostate_info.npy"), allow_pickle=True).item()  # noqa
        assert isinstance(self._prostate_info, dict) and len(self._prostate_info) == 50

    def __getitem__(self, index) -> Tuple[List[Tensor], str, Tuple[str, str]]:
        images, filename = super().__getitem__(index)
        partition = self._get_partition(filename)
        scan_num = self._get_scan_name(filename)
        return images, filename, (partition, scan_num)

    def _get_partition(self, filename) -> str:
        # set partition
        max_len_given_group = self._prostate_info[self._get_scan_name(filename)]
        cutting_point = max_len_given_group // self.partition_num
        cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        return str(cur_index // (cutting_point + 1))


class ProstateMDDataset(ContrastDataset, _prostate_md):
    partition_num = 4

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        super().__init__(root_dir=root_dir, mode=mode, transforms=transforms)
        self._prostate_info: Dict[str, int] = \
            np.load(os.path.join(self._root_dir, "prostate_info.npy"), allow_pickle=True).item()  # noqa
        assert isinstance(self._prostate_info, dict) and len(self._prostate_info) == 32

    def __getitem__(self, index) -> Tuple[List[Tensor], str, Tuple[str, str]]:
        images, filename = super().__getitem__(index)
        partition = self._get_partition(filename)
        scan_num = self._get_scan_name(filename)
        return images, filename, (partition, scan_num)

    def _get_partition(self, filename) -> str:
        # set partition
        max_len_given_group = self._prostate_info[self._get_scan_name(filename)]
        cutting_point = max_len_given_group // self.partition_num
        cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        return str(cur_index // (cutting_point + 1))


class _mmWHSBase(ContrastDataset):
    partition_num = 8
    _get_scan_name: Callable[[str], str]

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        super().__init__(root_dir=root_dir, mode=mode, transforms=transforms, )

        self._meta_info = {"ct": np.load(str(Path(root_dir, "MMWHS", "meta_ct.npy")), allow_pickle=True).tolist(),
                           "mr": np.load(str(Path(root_dir, "MMWHS", "meta_mr.npy")), allow_pickle=True).tolist()}

    def __getitem__(self, index) -> Tuple[List[Tensor], str, Tuple[str, str]]:
        images, filename = super().__getitem__(index)  # noqa
        partition = self._get_partition(filename)
        scan_num = self._get_scan_name(filename)
        return images, filename, (partition, scan_num)

    def _get_partition(self, filename) -> str:
        # set partition
        max_len_given_group = self.get_meta()[self._get_scan_name(filename)]
        cutting_point = max_len_given_group // self.partition_num
        cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        return str(cur_index // (cutting_point + 1))

    def get_meta(self) -> Dict[str, int]:
        ...


class mmWHSMRDataset(_mmWHSBase, _mmmr):

    def get_meta(self):
        return self._meta_info["mr"]


class mmWHSCTDataset(_mmWHSBase, _mmct):
    def get_meta(self):
        return self._meta_info["ct"]


class SpleenDataset(ContrastDataset, _spleen):
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        super().__init__(root_dir=root_dir, mode=mode, transforms=transforms)
        self._spleen_info: Dict[str, int] \
            = np.load(os.path.join(self._root_dir, "spleen_info.npy"), allow_pickle=True).item()  # noqa
        assert isinstance(self._spleen_info, dict) and len(self._spleen_info) == 41

    def __getitem__(self, index) -> Tuple[List[Tensor], str, Tuple[str, str]]:
        images, filename = super().__getitem__(index)
        partition = self._get_partition(filename)
        scan_num = self._get_scan_name(filename)
        return images, filename, (partition, scan_num)

    def _get_partition(self, filename) -> str:
        # set partition
        max_len_given_group = self._spleen_info[self._get_scan_name(filename)]
        cutting_point = max_len_given_group // self.partition_num
        cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        return str(cur_index // (cutting_point + 1))


class HippocampusDataset(ContrastDataset, _Hippocampus):
    partition_num = 3

    def __init__(self, *, root_dir: str, mode: str, transforms: SequentialWrapper = None) -> None:
        super().__init__(root_dir=root_dir, mode=mode, transforms=transforms)
        self._hippocampus_info: Dict[str, int] \
            = np.load(os.path.join(self._root_dir, "hippocampus_info.npy"), allow_pickle=True).item()  # noqa
        assert isinstance(self._hippocampus_info, dict) and len(self._hippocampus_info) == 260

    def __getitem__(self, index) -> Tuple[List[Tensor], str, Tuple[str, str]]:
        images, filename = super().__getitem__(index)
        partition = self._get_partition(filename)
        scan_num = self._get_scan_name(filename)
        return images, filename, (partition, scan_num)

    def _get_partition(self, filename) -> str:
        # set partition
        max_len_given_group = self._hippocampus_info[self._get_scan_name(filename)]
        cutting_point = max_len_given_group // self.partition_num
        cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        if cur_index <= cutting_point - 1:
            return str(0)
        if cur_index <= 2 * cutting_point:
            return str(1)
        return str(2)
