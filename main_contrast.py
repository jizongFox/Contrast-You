from scipy.sparse import issparse  # noqa

_ = issparse  # noqa
from pathlib import Path

from contrastyou import DATA_PATH, PROJECT_PATH
from contrastyou.arch import UNet
from contrastyou.augment import transform_dict
from contrastyou.dataloader._seg_datset import ContrastBatchSampler  # noqa
from contrastyou.dataloader.acdc_dataset import ACDCSemiInterface, ACDCDataset
from contrastyou.trainer import trainer_zoos
from deepclustering2.configparser import ConfigManger
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.dataset import PatientSampler
from deepclustering2.utils import gethash, set_benchmark
from torch.utils.data import DataLoader

# load configure from yaml and argparser
cmanager = ConfigManger(Path(PROJECT_PATH) / "config/config.yaml")
config = cmanager.config
cur_githash = gethash(__file__)

# set reproducibility
set_benchmark(1)

model = UNet(**config["Arch"])

acdc_manager = ACDCSemiInterface(root_dir=DATA_PATH, labeled_data_ratio=config["Data"]["labeled_data_ratio"],
                                 unlabeled_data_ratio=config["Data"]["unlabeled_data_ratio"])
transform = transform_dict[config.get("Augment", "simple")]
label_set, _, val_set = acdc_manager._create_semi_supervised_datasets(  # noqa
    labeled_transform=transform.label,
    val_transform=transform.val
)
train_set = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=transform.pretrain)

# all training set is with ContrastBatchSampler
train_loader = DataLoader(  # noqa
    train_set,  # noqa
    batch_sampler=ContrastBatchSampler(
        train_set,
        group_sample_num=config["ContrastData"]["group_sample_num"],
        partition_sample_num=config["ContrastData"]["partition_sample_num"],
        shuffle=config["ContrastData"]["shuffle"]
    ),
    num_workers=config["ContrastData"]["num_workers"],
    pin_memory=True
)

# labeled loader is with normal 2d slicing and InfiniteRandomSampler
labeled_loader = DataLoader(
    label_set, sampler=InfiniteRandomSampler(
        label_set,
        shuffle=config["LabeledData"]["shuffle"]
    ),
    batch_size=config["LabeledData"]["batch_size"],
    num_workers=config["LabeledData"]["num_workers"],
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_sampler=PatientSampler(
        val_set,
        grp_regex=val_set.dataset_pattern,
        shuffle=False
    ),
    pin_memory=True
)

checkpoint = config.pop("Checkpoint", None)
Trainer = trainer_zoos[config["Trainer"].pop("name")]
assert Trainer, Trainer
trainer = Trainer(model=model, pretrain_loader=train_loader, fine_tune_loader=labeled_loader, val_loader=val_loader,
                  configuration={**cmanager.config, **{"GITHASH": cur_githash}},
                  **config["Trainer"], )

trainer.start_training(checkpoint=checkpoint, pretrain_encoder_init_options=config["PretrainEncoder"],
                       pretrain_decoder_init_options=config["PretrainDecoder"],
                       finetune_network_init_options=config["FineTune"])
