from pathlib import Path

from torch.utils.data import DataLoader

from contrastyou import DATA_PATH, PROJECT_PATH
from contrastyou.augment import ACDC_transforms
from contrastyou.dataloader._seg_datset import ContrastBatchSampler
from contrastyou.dataloader.acdc_dataset import ACDCSemiInterface, ACDCDataset
from contrastyou.losses.iic_loss import IIDSegmentationLoss
from contrastyou.trainer.contrast_trainer import trainer_zoos
from deepclustering2.configparser import ConfigManger
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.dataset import PatientSampler
from deepclustering2.loss import KL_div
from deepclustering2.models import Model
from deepclustering2.utils import set_benchmark

set_benchmark(1)

cmanager = ConfigManger(Path(PROJECT_PATH) / "config/config.yaml")
config = cmanager.config

acdc_manager = ACDCSemiInterface(root_dir=DATA_PATH, labeled_data_ratio=0.95, unlabeled_data_ratio=0.05)

label_set, unlabel_set, val_set = acdc_manager._create_semi_supervised_datasets(
    labeled_transform=ACDC_transforms.train,
    unlabeled_transform=ACDC_transforms.train,
    val_transform=ACDC_transforms.val
)
train_set = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=ACDC_transforms.train)

train_loader = DataLoader(train_set, sampler=InfiniteRandomSampler(train_set), num_workers=4, pin_memory=True)

labeled_loader = DataLoader(label_set,
                            batch_sampler=ContrastBatchSampler(label_set, group_sample_num=4, partition_sample_num=1),
                            num_workers=4, pin_memory=True)
unlabeled_loader = DataLoader(unlabel_set,
                              batch_sampler=ContrastBatchSampler(unlabel_set, group_sample_num=4,
                                                                 partition_sample_num=1),
                              num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set,
                        batch_sampler=PatientSampler(
                            val_set,
                            grp_regex=val_set.dataset_pattern,
                            shuffle=False),
                        pin_memory=True)

reg_criterion = IIDSegmentationLoss(padding=5)

model = Model(config["Arch"], config["Optim"], config["Scheduler"])
trainer_name = config["Trainer"].pop("name")
Trainer = trainer_zoos.get(trainer_name)
assert Trainer

trainer = Trainer(model, iter(train_loader), iter(labeled_loader), iter(unlabeled_loader), val_loader,
                  sup_criterion=KL_div(), reg_criterion=reg_criterion, configuration=cmanager.config,
                  **config["Trainer"])

trainer.start_training()
