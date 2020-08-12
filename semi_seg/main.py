import os
from itertools import chain

from deepclustering2.loss import KL_div
from scipy.sparse import issparse  # noqa

_ = issparse  # noqa
from pathlib import Path
from contrastyou import DATA_PATH, PROJECT_PATH
from contrastyou.arch import UNet
from contrastyou.augment import transform_dict
from contrastyou.dataloader._seg_datset import ContrastBatchSampler  # noqa
from contrastyou.dataloader.acdc_dataset import ACDCSemiInterface
from deepclustering2.configparser import ConfigManger
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.dataset import PatientSampler
from deepclustering2.utils import gethash
from torch.utils.data import DataLoader
from deepclustering2.utils import set_benchmark
from semi_seg.trainer import trainer_zoos
from semi_seg._utils import LocalClusterWrappaer
from deepclustering2 import optim
from contrastyou.losses.iic_loss import IIDSegmentationSmallPathLoss

# load configure from yaml and argparser
cmanager = ConfigManger(Path(PROJECT_PATH) / "config/semi.yaml")
config = cmanager.config
cur_githash = gethash(__file__)

# set reproducibility
set_benchmark(1)

acdc_manager = ACDCSemiInterface(root_dir=DATA_PATH, labeled_data_ratio=config["Data"]["labeled_data_ratio"],
                                 unlabeled_data_ratio=config["Data"]["unlabeled_data_ratio"])
transform = transform_dict[config.get("Augment", "simple")]
label_set, unlabel_set, val_set = acdc_manager._create_semi_supervised_datasets(  # noqa
    labeled_transform=transform.label,
    unlabeled_transform=transform.pretrain,
    val_transform=transform.val
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
unlabeled_loader = DataLoader(
    unlabel_set, sampler=InfiniteRandomSampler(
        unlabel_set,
        shuffle=config["UnlabeledData"]["shuffle"]
    ),
    batch_size=config["UnlabeledData"]["batch_size"],
    num_workers=config["UnlabeledData"]["num_workers"],
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
trainer_name = config["Trainer"].pop("name")
Trainer = trainer_zoos[trainer_name]

model = UNet(**config["Arch"])
optimizer = optim.__dict__[config["Optim"]["name"]](
    params=model.parameters(),
    **{k: v for k, v in config["Optim"].items() if k != "name"}
)
if "iic" in trainer_name:
    projectors_wrapper = LocalClusterWrappaer(**config["LocalCluster"])

    optimizer = optim.__dict__[config["Optim"]["name"]](
        params=chain(model.parameters(), projectors_wrapper.parameters()),
        **{k: v for k, v in config["Optim"].items() if k != "name"}
    )
    IIDSegCriterion = IIDSegmentationSmallPathLoss(**config["IIDSeg"])
    Trainer.set_feature_positions(config["LocalCluster"]["feature_names"])
    trainer = Trainer(model=model, optimizer=optimizer, labeled_loader=iter(labeled_loader),
                      unlabeled_loader=iter(unlabeled_loader),
                      val_loader=val_loader,
                      sup_criterion=KL_div(),
                      configuration={**cmanager.config, **{"GITHASH": cur_githash}},
                      projector_wrappers=projectors_wrapper,
                      IIDSegCriterion=IIDSegCriterion,
                      **config["Trainer"], )
else:
    trainer = Trainer(model=model, optimizer=optimizer, labeled_loader=iter(labeled_loader),
                      unlabeled_loader=iter(unlabeled_loader),
                      val_loader=val_loader,
                      sup_criterion=KL_div(),
                      configuration={**cmanager.config, **{"GITHASH": cur_githash}},
                      **config["Trainer"], )
checkpoint = config.get("Checkpoint", None)
if checkpoint is not None:
    trainer.load_state_dict_from_path(os.path.join(checkpoint, "last.pth"))
trainer.start_training()

