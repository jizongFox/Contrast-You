from copy import deepcopy as dcopy
from pathlib import Path

from loguru import logger
from torch.utils.data.dataloader import _BaseDataLoaderIter as BaseDataLoaderIter, DataLoader  # noqa

from contrastyou import get_true_data_path
from contrastyou.data import InfiniteRandomSampler, ScanBatchSampler, DatasetBase
from contrastyou.utils import get_dataset, fix_seed, class_name
from semi_seg.data import ACDCDataset, ProstateDataset, mmWHSCTDataset, ProstateMDDataset, SpleenDataset, \
    HippocampusDataset
from semi_seg.data.creator import augment_zoo, data_zoo
from semi_seg.data.rearr import ContrastBatchSampler


def get_partition_num(name: str):
    partition_num_zoo = {"acdc": ACDCDataset.partition_num,
                         "acdc_lv": ACDCDataset.partition_num,
                         "acdc_rv": ACDCDataset.partition_num,
                         "prostate": ProstateDataset.partition_num,
                         "mmwhsct": mmWHSCTDataset.partition_num,
                         "mmwhsmr": mmWHSCTDataset.partition_num,
                         "prostate_md": ProstateMDDataset.partition_num,
                         "spleen": SpleenDataset.partition_num,
                         "hippocampus": HippocampusDataset.partition_num}
    return partition_num_zoo[name]


@fix_seed
def _get_contrastive_dataloader(partial_loader, contrastive_params):
    contrastive_params = dcopy(contrastive_params)
    # going to get all dataset with contrastive sampler
    dataset: DatasetBase = get_dataset(partial_loader)
    data_name = {class_.__name__: name for name, class_ in data_zoo.items()}[dataset._name.split("-")[0]]
    is_preload = dataset._is_preload  # noqa
    dataset_type = dataset.__class__
    dataset = dataset_type(root_dir=get_true_data_path(), mode="train", transforms=dcopy(dataset.transforms))  # noqa

    logger.opt(depth=2).debug(f"creating {dataset.__class__.__name__} contrastive dataset with "
                              f"{len(dataset.get_scan_list())} scans")

    if is_preload:
        dataset.preload()

    num_workers = contrastive_params.pop("num_workers")

    augment = augment_zoo[data_name]

    batch_sampler = None

    batch_size = contrastive_params["scan_sample_num"] * get_partition_num(data_name)

    from semi_seg import PRETRAIN_BATCH_SIZE_MAX
    batch_size = min(batch_size, PRETRAIN_BATCH_SIZE_MAX)

    sampler = InfiniteRandomSampler(dataset, shuffle=True)

    logger.opt(depth=2).trace(f"Contrastive {class_name(dataset)} "
                              f"with batch_size = {batch_size}, num_workers = {num_workers} ")

    if data_name in ("acdc", "spleen"):
        # only group the acdc dataset
        batch_sampler = ContrastBatchSampler(dataset=dataset, **contrastive_params)
        logger.opt(depth=2).trace(f"{data_name} created contrastive batch sampler")
        batch_size = 1
        sampler = None

    contrastive_loader = DataLoader(
        dataset, batch_sampler=batch_sampler, sampler=sampler,
        num_workers=num_workers, batch_size=batch_size,
        pin_memory=True, shuffle=False,
    )

    demo_dataset = dataset_type(root_dir=str(Path(dataset._root_dir).parent),  # noqa
                                mode="train", transforms=augment.trainval)

    demo_loader = DataLoader(demo_dataset, batch_size=1, batch_sampler=ScanBatchSampler(dataset))
    logger.opt(depth=2).trace(f"creating {dataset.__class__.__name__} demo dataset with {len(dataset)} images")
    return contrastive_loader, demo_loader
