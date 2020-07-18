from pathlib import Path

from torch.utils.data import DataLoader

from contrastyou import DATA_PATH, PROJECT_PATH
from contrastyou.arch import UNet
from contrastyou.augment import ACDC_transforms
from contrastyou.dataloader._seg_datset import ContrastBatchSampler  # noqa
from contrastyou.dataloader.acdc_dataset import ACDCSemiInterface, ACDCDataset
from contrastyou.trainer import ContrastTrainer
from deepclustering2.configparser import ConfigManger
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.dataset import PatientSampler
from deepclustering2.utils import set_benchmark

set_benchmark(1)

cmanager = ConfigManger(Path(PROJECT_PATH) / "config/config.yaml")
config = cmanager.config

acdc_manager = ACDCSemiInterface(root_dir=DATA_PATH, labeled_data_ratio=config["Data"]["labeled_data_ratio"],
                                 unlabeled_data_ratio=config["Data"]["unlabeled_data_ratio"])

label_set, unlabel_set, val_set = acdc_manager._create_semi_supervised_datasets(  # noqa
    labeled_transform=ACDC_transforms.train,
    unlabeled_transform=ACDC_transforms.train,
    val_transform=ACDC_transforms.val
)
train_set = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=ACDC_transforms.train)

train_loader = DataLoader(train_set, batch_sampler=ContrastBatchSampler(train_set, group_sample_num=4,
                                                                        partition_sample_num=1),
                          num_workers=8, pin_memory=True, )

labeled_loader = DataLoader(label_set, sampler=InfiniteRandomSampler(label_set, shuffle=True), batch_size=32,
                            num_workers=8, pin_memory=True)

val_loader = DataLoader(val_set, batch_sampler=PatientSampler(
    val_set,
    grp_regex=val_set.dataset_pattern,
    shuffle=False), pin_memory=True)

model = UNet(input_dim=1, num_classes=4)

checkpoint = config["Trainer"].pop("checkpoint", None)

trainer = ContrastTrainer(model, iter(train_loader), iter(labeled_loader), val_loader,
                          configuration=cmanager.config, **config["Trainer"], max_epoch_train_decoder=100,
                          max_epoch_train_encoder=100, max_epoch_train_finetune=100)
if checkpoint:
    trainer.load_state_dict_from_path(checkpoint)

trainer.start_training()
