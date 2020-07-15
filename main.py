from pathlib import Path

import torch
from torch.utils.data import DataLoader

from contrastyou import DATA_PATH, PROJECT_PATH
from contrastyou.augment import ACDC_transforms
from contrastyou.dataloader._seg_datset import ContrastBatchSampler
from contrastyou.dataloader.acdc_dataset import ACDCSemiInterface, ACDCDataset
from contrastyou.trainer import trainer_zoos
from deepclustering2.configparser import ConfigManger
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.dataset import PatientSampler
from deepclustering2.loss import KL_div
from deepclustering2.models import Model
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

if config["Data"]["use_contrast"]:
    train_loader = DataLoader(train_set, batch_sampler=ContrastBatchSampler(train_set, group_sample_num=8,
                                                                            partition_sample_num=1),
                              num_workers=8, pin_memory=True, )

    labeled_loader = DataLoader(label_set,
                                batch_sampler=ContrastBatchSampler(label_set, group_sample_num=4,  # noqa
                                                                   partition_sample_num=1),
                                num_workers=8, pin_memory=True)
    unlabeled_loader = DataLoader(unlabel_set,
                                  batch_sampler=ContrastBatchSampler(unlabel_set, group_sample_num=4, # noqa
                                                                     partition_sample_num=1),
                                  num_workers=4, pin_memory=True)

else:
    train_loader = DataLoader(train_set, sampler=InfiniteRandomSampler(train_set, shuffle=True), num_workers=8,
                              pin_memory=True,
                              batch_size=8, )
    labeled_loader = DataLoader(label_set,
                                sampler=InfiniteRandomSampler(label_set, shuffle=True),
                                num_workers=4, pin_memory=True, batch_size=4)
    unlabeled_loader = DataLoader(unlabel_set,
                                  sampler=InfiniteRandomSampler(unlabel_set, shuffle=True),
                                  num_workers=4, pin_memory=True, batch_size=4)

val_loader = DataLoader(val_set, batch_sampler=PatientSampler(
    val_set,
    grp_regex=val_set.dataset_pattern,
    shuffle=False), pin_memory=True)


def NullLoss(pred, *args, **kwargs):
    return torch.tensor(0.0, dtype=torch.float, device=pred.device)


reg_criterion = NullLoss

model = Model(config["Arch"], config["Optim"], config["Scheduler"])
trainer_name = config["Trainer"].pop("name", None)
Trainer = trainer_zoos.get(trainer_name)
assert Trainer, trainer_name
checkpoint = config["Trainer"].pop("checkpoint", None)

trainer = Trainer(model, iter(train_loader), iter(labeled_loader), iter(unlabeled_loader), val_loader,
                  sup_criterion=KL_div(), reg_criterion=reg_criterion, configuration=cmanager.config,
                  **config["Trainer"])
if checkpoint:
    trainer.load_state_dict_from_path(checkpoint)

trainer.start_training()
