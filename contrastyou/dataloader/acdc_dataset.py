from typing import List, Tuple, Union

from deepclustering.augment import SequentialWrapper
from deepclustering.dataset import ACDCDataset as _ACDCDataset
from torch import Tensor

from contrastyou.dataloader._seg_datset import ContrastDataset


class ACDCDataset(ContrastDataset, _ACDCDataset):
    def __init__(self, root_dir: str, mode: str, transforms: SequentialWrapper = None,
                 verbose=True) -> None:
        super().__init__(root_dir, mode, ["img", "gt"], transforms, verbose)

    def __getitem__(self, index) -> Tuple[List[Tensor], str, str, str]:
        data, filename = super().__getitem__(index)
        partition = self._get_partition(filename)
        group = self._get_group(filename)
        return data, filename, partition, group

    def _get_group(self, filename) -> Union[str, int]:
        return self._get_group_name(filename)

    def _get_partition(self, filename) -> Union[str, int]:
        return 0

    def show_paritions(self) -> List[Union[str, int]]:
        return [self._get_partition(f) for f in list(self._filenames.values())[0]]

    def show_groups(self) -> List[Union[str, int]]:
        return [self._get_group(f) for f in list(self._filenames.values())[0]]




