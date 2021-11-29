import random
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from copy import deepcopy as dcopy
from typing import Union, List, Callable, Dict

from torch.utils.data.sampler import Sampler

from contrastyou.data.dataset.base import get_stem

__all__ = ["ContrastDataset", "ContrastBatchSampler"]


class ContrastDataset(metaclass=ABCMeta):
    """
    each patient has 2 code, the first code is the group_name, which is the patient id
    the second code is the partition code, indicating the position of the image slice.
    All patients should have the same partition numbers so that they can be aligned.
    For ACDC dataset, the ED and ES ventricular volume should be considered further
    """
    get_memory_dictionary: Callable[[], Dict[str, List[str]]]
    __len__: Callable[[], int]

    @abstractmethod
    def _get_partition(self, *args) -> Union[str, int]:
        """get the partition of a 2D slice given its index or filename"""
        pass

    def get_partition_list(self) -> List[Union[str, int]]:
        """show all groups of 2D slices in the dataset"""

        return [self._get_partition(get_stem(f)) for f in next(iter(self.get_memory_dictionary().values()))]


class ContrastBatchSampler(Sampler):
    """
    This class is going to realize the sampling for different patients and from the same patients
    `we form batches by first randomly sampling m < M volumes. Then, for each sampled volume, we sample one image per
    partition resulting in S images per volume. Next, we apply a pair of random transformations on each sampled image and
    add them to the batch
    """

    class _SamplerIterator:

        def __init__(self, scan2index, partition2index, scan_sample_num=4, partition_sample_num=1,
                     shuffle=False) -> None:
            self._group2index, self._partition2index = dcopy(scan2index), dcopy(partition2index)

            assert 1 <= scan_sample_num <= len(self._group2index.keys()), scan_sample_num
            self._scan_sample_num = scan_sample_num
            self._partition_sample_num = partition_sample_num
            self._shuffle = shuffle

        def __iter__(self):
            return self

        def __next__(self):
            batch_index = []
            cur_group_samples = random.sample(list(self._group2index.keys()), self._scan_sample_num)
            assert isinstance(cur_group_samples, list), cur_group_samples

            # for each group sample, choose at most partition_sample_num slices per partition
            for cur_group in cur_group_samples:
                available_slices_given_group = self._group2index[cur_group]
                for s_available_slices in self._partition2index.values():
                    try:
                        sampled_slices = random.sample(
                            sorted(set(available_slices_given_group) & set(s_available_slices)),
                            self._partition_sample_num
                        )
                        batch_index.extend(sampled_slices)
                    except ValueError:
                        continue
            if self._shuffle:
                random.shuffle(batch_index)
            return batch_index

    def __init__(self, dataset: ContrastDataset, scan_sample_num=4, partition_sample_num=1, shuffle=False) -> None:
        super(ContrastBatchSampler, self).__init__(data_source=dataset)
        self._dataset = dataset
        filenames = dcopy(next(iter(dataset.get_memory_dictionary().values())))
        scan2index, partition2index = defaultdict(lambda: []), defaultdict(lambda: [])

        for i, filename in enumerate(filenames):
            group = dataset._get_scan_name(filename)  # noqa
            scan2index[group].append(i)
            partition = dataset._get_partition(filename)  # noqa
            partition2index[partition].append(i)

        self._scan2index = scan2index
        self._partition2index = partition2index
        self._scan_sample_num = scan_sample_num
        self._partition_sample_num = partition_sample_num
        self._shuffle = shuffle

    def __iter__(self):
        return self._SamplerIterator(self._scan2index, self._partition2index, self._scan_sample_num,
                                     self._partition_sample_num, shuffle=self._shuffle)

    def __len__(self) -> int:
        return len(self._dataset)
