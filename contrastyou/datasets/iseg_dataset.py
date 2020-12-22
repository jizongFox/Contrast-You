from typing import List

from deepclustering2.augment import SequentialWrapper
from deepclustering2.dataset.segmentation.iSeg2017_dataset import ISeg2017Dataset as _iSegDataset, \
    ISeg2017SemiInterface as _iSegSemiInterface
from ._seg_datset import ContrastDataset


class iSegDataset(ContrastDataset, _iSegDataset):

    def __init__(self, root_dir: str, mode: str, subfolders: List[str],
                 transforms: SequentialWrapper = SequentialWrapper(),
                 verbose=False) -> None:
        super().__init__(root_dir, mode, subfolders, transforms, verbose)
        self._transform = transforms
