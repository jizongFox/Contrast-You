import json
import os
from typing import Tuple, List

import numpy as np
from loguru import logger
from torch.utils.data import DataLoader

from contrastyou import get_cc_data_path
from contrastyou.data import DatasetBase, extract_sub_dataset_based_on_scan_names, InfiniteRandomSampler, \
    ScanBatchSampler
from contrastyou.utils import fix_all_seed_within_context
from semi_seg.augment import ACDCStrongTransforms, SpleenStrongTransforms, ProstateStrongTransforms, \
    HippocampusStrongTransforms
from semi_seg.data import ACDCDataset, ProstateDataset, mmWHSCTDataset, mmWHSMRDataset, ProstateMDDataset, \
    SpleenDataset, HippocampusDataset

data_zoo = {"acdc": ACDCDataset, "prostate": ProstateDataset, "prostate_md": ProstateMDDataset,
            "mmwhsct": mmWHSCTDataset, "mmwhsmr": mmWHSMRDataset, "spleen": SpleenDataset,
            "hippocampus": HippocampusDataset}
augment_zoo = {
    "acdc": ACDCStrongTransforms, "spleen": SpleenStrongTransforms,
    "prostate": ProstateStrongTransforms, "mmwhsct": ACDCStrongTransforms, "mmwhsmr": ACDCStrongTransforms,
    "prostate_md": ProstateStrongTransforms, "hippocampus": HippocampusStrongTransforms,
}

__all__ = ["create_dataset", "create_val_loader", "get_data_loaders", "get_data"]


def create_dataset(name: str, total_freedom: bool = True):
    data_class = data_zoo[name]
    aug_transform = augment_zoo[name]
    tra_transform = aug_transform.pretrain
    tra_transform._total_freedom = total_freedom
    tra_set = data_class(root_dir=get_cc_data_path(), mode="train", transforms=tra_transform)
    test_set = data_class(root_dir=get_cc_data_path(), mode="val", transforms=aug_transform.val)
    assert set(tra_set.get_scan_list()) & set(test_set.get_scan_list()) == set()
    return tra_set, test_set


def split_dataset_with_predefined_filenames(dataset: DatasetBase, data_name: str, labeled_scan_nums: int, order=0):
    try:
        with open(os.path.join(dataset._root_dir, f"{data_name}_ordering_{order}.json"), "r") as f:  # noqa
            data_ordering: List[str] = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{data_name} does not have {data_name}_ordering.json to predefine the ordering.")

    assert set(dataset.get_scan_list()) == set(data_ordering), \
        "dataset inconsistency between ordering.json and the set."
    assert len(dataset.get_scan_list()) >= labeled_scan_nums, f"labeled_scan_nums greater than the dataset"

    labeled_scans = data_ordering[:labeled_scan_nums]
    unlabeled_scans = sorted(set(dataset.get_scan_list()) - set(labeled_scans))
    logger.debug(f"using predefined ordering to {dataset.__class__.__name__}.")
    return [extract_sub_dataset_based_on_scan_names(dataset, group_names=labeled_scans),
            extract_sub_dataset_based_on_scan_names(dataset, group_names=unlabeled_scans)]


def split_dataset(dataset: DatasetBase, *scans: float, seed: int = 1) -> List[DatasetBase]:
    """this function splits the dataset without considering the predefined ordering, controlled by a seed"""
    assert sum(scans) <= len(dataset.get_scan_list()), scans
    scan_list = sorted(set(dataset.get_scan_list()))
    with fix_all_seed_within_context(seed):
        scan_list_permuted = np.random.permutation(scan_list).tolist()

    def _sum_iter(ratio_list):
        sum = 0
        for i in ratio_list:
            yield sum + i
            sum += i

    def _two_element_iter(cut_list):
        previous = 0
        for r in cut_list:
            yield previous, r
            previous = r
        yield previous, len(scan_list)

    cutting_points = [x for x in _sum_iter(scans)]

    sub_datasets = [extract_sub_dataset_based_on_scan_names(dataset, scan_list_permuted[x:y]) for x, y in
                    _two_element_iter(cutting_points)]
    assert sum([len(set(x.get_scan_list())) for x in sub_datasets]) == len(scan_list)
    return sub_datasets


def create_infinite_loader(dataset, shuffle=True, num_workers: int = 8, batch_size: int = 4):
    sampler = InfiniteRandomSampler(dataset, shuffle=shuffle)

    loader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    return loader


def get_data_loaders(data_params, labeled_loader_params, unlabeled_loader_params, pretrain=False, group_test=True,
                     total_freedom=False):
    data_name = data_params["name"]
    tra_set, test_set = create_dataset(data_name, total_freedom)
    if len(tra_set.get_scan_list()) == 0 or len(test_set.get_scan_list()) == 0:
        raise RuntimeError("dataset loading error with empty dataset ont `tra_set` and `val_set`")

    train_scan_num = len(tra_set.get_scan_list())

    labeled_scan_num = int(data_params["labeled_scan_num"])
    if labeled_scan_num > train_scan_num:
        raise RuntimeError(f"labeled scan number {labeled_scan_num} greater than the train set size: {train_scan_num}")

    if pretrain:
        labeled_scan_num = int(train_scan_num // 2)
        label_set, unlabeled_set = split_dataset(tra_set, labeled_scan_num)
    else:
        try:
            label_set, unlabeled_set = split_dataset_with_predefined_filenames(tra_set, data_name,
                                                                               labeled_scan_nums=labeled_scan_num)
        except FileNotFoundError as e:
            seed = 2
            logger.critical(f"{data_name} did not find the ordering json file, "
                            f"using a random split with random seed: {seed}")
            logger.critical(e)
            label_set, unlabeled_set = split_dataset(tra_set, labeled_scan_num, seed=seed)

    if len(label_set.get_scan_list()) == 0:
        raise RuntimeError("void labeled dataset, split dataset error")

    labeled_loader = create_infinite_loader(label_set, **labeled_loader_params)
    logger.debug(f"creating labeled_loader with {len(label_set.get_scan_list())} scans")
    logger.trace(f"with {','.join(sorted(set(label_set.get_scan_list())))}")

    unlabeled_loader = create_infinite_loader(unlabeled_set, **unlabeled_loader_params)
    logger.debug(f"creating unlabeled_loader with {len(unlabeled_set.get_scan_list())} scans")
    logger.trace(f"with {','.join(sorted(set(unlabeled_set.get_scan_list())))}")

    group_test = group_test if data_name not in (
    "spleen", "mmwhsct", "mmwhsmr", "prostate_md", "hippocampus") else False
    test_loader = DataLoader(test_set, batch_size=1 if group_test else 4,
                             batch_sampler=ScanBatchSampler(test_set, shuffle=False) if group_test else None)

    return labeled_loader, unlabeled_loader, test_loader


def create_val_loader(*, test_loader) -> Tuple[DataLoader, DataLoader]:
    test_dataset: DatasetBase = test_loader.dataset
    test_size = len(test_dataset.get_scan_list())
    batch_sampler = test_loader.batch_sampler
    is_group_scan = isinstance(batch_sampler, ScanBatchSampler)

    if_mmwhs_dataset = isinstance(test_dataset, (mmWHSCTDataset, mmWHSMRDataset))

    scan_nums = int(0.35 * test_size) if not if_mmwhs_dataset else int(0.45 * test_size)
    val_set, test_set = split_dataset(test_dataset, scan_nums)
    val_batch_sampler = ScanBatchSampler(val_set) if is_group_scan else None

    val_dataloader = DataLoader(val_set, batch_sampler=val_batch_sampler, batch_size=4 if not is_group_scan else 1)

    test_batch_sampler = ScanBatchSampler(test_set) if is_group_scan else None
    test_dataloader = DataLoader(test_set, batch_sampler=test_batch_sampler, batch_size=4 if not is_group_scan else 1)

    logger.debug(f"splitting val_loader with {len(val_set.get_scan_list())} scans")
    logger.trace(f" with {','.join(sorted(set(val_set.get_scan_list())))}")
    logger.debug(f"splitting test_loader with {len(test_set.get_scan_list())} scans")
    logger.trace(f"with {','.join(sorted(set(test_set.get_scan_list())))}")

    return val_dataloader, test_dataloader


def get_data(data_params, labeled_loader_params, unlabeled_loader_params, pretrain=False, total_freedom=False):
    labeled_loader, unlabeled_loader, test_loader = get_data_loaders(
        data_params=data_params, labeled_loader_params=labeled_loader_params,
        unlabeled_loader_params=unlabeled_loader_params, pretrain=pretrain, group_test=True, total_freedom=total_freedom
    )
    val_loader, test_loader = create_val_loader(test_loader=test_loader)
    return labeled_loader, unlabeled_loader, val_loader, test_loader
