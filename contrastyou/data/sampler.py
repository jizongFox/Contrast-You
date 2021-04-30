import random
from collections import Iterator, defaultdict
from typing import List

import torch
from loguru import logger
from torch._six import int_classes as _int_classes
from torch.utils.data import Sampler


class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):  # noqa
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        super(RandomSampler, self).__init__(data_source)
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

        if self._num_samples is not None and not replacement:
            raise ValueError(
                "With replacement=False, num_samples should not be specified, "
                "since a random permute will be performed."
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(
                torch.randint(
                    high=n, size=(self.num_samples,), dtype=torch.int64
                ).tolist()
            )
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples


class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [0, 0, 0, 1, 0]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """

    def __init__(self, weights, num_samples, replacement=True):
        if (
            not isinstance(num_samples, _int_classes)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(num_samples)
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(replacement)
            )
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(
            torch.multinomial(self.weights, self.num_samples, self.replacement).tolist()
        )

    def __len__(self):
        return self.num_samples


class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        if (
            not isinstance(batch_size, _int_classes)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class InfiniteRandomSampler(Sampler):
    class _InfiniteRandomIterator(Iterator):
        def __init__(self, data_source, shuffle=True):
            self.data_source = data_source
            self.shuffle = shuffle
            if self.shuffle:
                self.iterator = iter(torch.randperm(len(self.data_source)).tolist())
            else:
                self.iterator = iter(
                    torch.arange(start=0, end=len(self.data_source)).tolist()
                )

        def __next__(self):
            try:
                idx = next(self.iterator)
            except StopIteration:
                if self.shuffle:
                    self.iterator = iter(torch.randperm(len(self.data_source)).tolist())
                else:
                    self.iterator = iter(
                        torch.arange(start=0, end=len(self.data_source)).tolist()
                    )
                idx = next(self.iterator)
            return idx

    def __init__(self, data_source, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        return self._InfiniteRandomIterator(self.data_source, shuffle=self.shuffle)

    def __len__(self):
        return len(self.data_source)


class ScanSampler(Sampler):
    from .dataset.base import DatasetBase

    def __init__(self, dataset: DatasetBase, shuffle=False, is_infinite: bool = False) -> None:
        scan_names: List[str] = [dataset._get_scan_name(x) for x in dataset.get_stem_list()]
        assert len(scan_names) == len(dataset), (len(scan_names), len(dataset))
        self._shuffle: bool = shuffle
        self._shuffle_fn = (lambda x: random.sample(x, len(x))) if self._shuffle else (lambda x: x)
        self._infinite_sampler = is_infinite

        unique_scans: List[str] = sorted(set(scan_names))
        assert len(unique_scans) < len(scan_names)
        logger.trace(f"Found {len(unique_scans)} unique patients out of {len(scan_names)} images")
        self.idx_map = defaultdict(list)
        for i, patient in enumerate(scan_names):
            self.idx_map[patient] += [i]
        assert sum(len(self.idx_map[k]) for k in unique_scans) == len(scan_names)
        logger.trace("Scan to slices mapping done")

    def __len__(self):
        return len(self.idx_map.keys())

    def __iter__(self):
        if not self._infinite_sampler:
            return self._one_iter()
        return self._infinite_iter()

    def _one_iter(self):
        values = list(self.idx_map.values())
        shuffled = self._shuffle_fn(values)
        return iter(shuffled)

    def _infinite_iter(self):
        while True:
            yield from self._one_iter()
