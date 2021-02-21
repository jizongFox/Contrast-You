from copy import deepcopy

from loguru import logger
from torch.utils.data import DataLoader

from contrastyou import DATA_PATH
from contrastyou.datasets import ACDCSemiInterface, SpleenSemiInterface, ProstateSemiInterface, MMWHSSemiInterface
from deepclustering2.dataloader.distributed import InfiniteDistributedSampler
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.dataset import PatientSampler
from semi_seg.augment import ACDCStrongTransforms, SpleenStrongTransforms, ProstateStrongTransforms

dataset_zoos = {
    "acdc": ACDCSemiInterface,
    "spleen": SpleenSemiInterface,
    "prostate": ProstateSemiInterface,
    "mmwhs": MMWHSSemiInterface,
}
augment_zoos = {
    "acdc": ACDCStrongTransforms,
    "spleen": SpleenStrongTransforms,
    "prostate": ProstateStrongTransforms,
    "mmwhs": ACDCStrongTransforms
}


def get_dataloaders(config, group_val_patient=True):
    _config = deepcopy(config)
    dataset_name = _config["Data"].pop("name")
    logger.debug("Initializing {} dataset", dataset_name)
    assert dataset_name in dataset_zoos.keys(), config["Data"]
    datainterface = dataset_zoos[dataset_name]
    augmentinferface = augment_zoos[dataset_name]

    data_manager = datainterface(root_dir=DATA_PATH, labeled_data_ratio=config["Data"]["labeled_data_ratio"],
                                 unlabeled_data_ratio=config["Data"]["unlabeled_data_ratio"], verbose=False)

    label_set, unlabel_set, val_set = data_manager._create_semi_supervised_datasets(  # noqa
        labeled_transform=augmentinferface.pretrain,
        unlabeled_transform=augmentinferface.pretrain,
        val_transform=augmentinferface.val
    )

    # labeled loader is with normal 2d slicing and InfiniteRandomSampler
    labeled_sampler = InfiniteRandomSampler(
        label_set,
        shuffle=config["LabeledData"]["shuffle"]
    )
    unlabeled_sampler = InfiniteRandomSampler(
        unlabel_set,
        shuffle=config["UnlabeledData"]["shuffle"]
    )
    if config.get("DistributedTrain") is True:
        labeled_sampler = InfiniteDistributedSampler(
            label_set,
            shuffle=config["LabeledData"]["shuffle"]
        )
        unlabeled_sampler = InfiniteDistributedSampler(
            unlabel_set,
            shuffle=config["UnlabeledData"]["shuffle"]
        )

    labeled_loader = DataLoader(
        label_set, sampler=labeled_sampler,
        batch_size=config["LabeledData"]["batch_size"],
        num_workers=config["LabeledData"]["num_workers"],
        pin_memory=True
    )
    unlabeled_loader = DataLoader(
        unlabel_set, sampler=unlabeled_sampler,
        batch_size=config["UnlabeledData"]["batch_size"],
        num_workers=config["UnlabeledData"]["num_workers"],
        pin_memory=True
    )
    group_val_patient = group_val_patient if dataset_name not in ("spleen", "mmwhs") else False
    if group_val_patient:
        logger.debug("grouping val patients")
    val_loader = DataLoader(
        val_set,
        batch_size=1 if group_val_patient else 4,
        batch_sampler=PatientSampler(
            val_set,
            grp_regex=val_set.dataset_pattern,
            shuffle=False
        ) if group_val_patient else None,
    )
    return labeled_loader, unlabeled_loader, val_loader
