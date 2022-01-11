from deepclustering2.dataset import ACDCSemiInterface

from contrastyou import DATA_PATH
from semi_seg.augment import ACDCTransforms2

datainterface_zoos = {"acdc": ACDCSemiInterface,
                      }
augment_zoos = {"acdc": ACDCTransforms2()
                }


def create_dataset(name="acdc", labeled_ratio=0.1, **kwargs):
    interface = datainterface_zoos[name]
    augment = augment_zoos[name]
    manager = interface(root_dir=DATA_PATH, labeled_data_ratio=labeled_ratio,
                        unlabeled_data_ratio=1 - labeled_ratio, **kwargs)
    label_set, unlabel_set, val_set = manager._create_semi_supervised_datasets(  # noqa
        labeled_transform=augment.pretrain,
        unlabeled_transform=augment.pretrain,
        val_transform=augment.val
    )
    return label_set, unlabel_set, val_set
