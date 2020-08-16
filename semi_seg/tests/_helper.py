from contrastyou import DATA_PATH
from contrastyou.augment import ACDCStrongTransforms
from contrastyou.dataloader.acdc_dataset import ACDCSemiInterface


def create_acdc_dataset(labeled_ratio=0.1):
    acdc_manager = ACDCSemiInterface(root_dir=DATA_PATH, labeled_data_ratio=labeled_ratio,
                                     unlabeled_data_ratio=1 - labeled_ratio)
    label_set, unlabel_set, val_set = acdc_manager._create_semi_supervised_datasets(  # noqa
        labeled_transform=ACDCStrongTransforms.label,
        unlabeled_transform=ACDCStrongTransforms.pretrain,
        val_transform=ACDCStrongTransforms.val
    )
    return label_set, unlabel_set, val_set
