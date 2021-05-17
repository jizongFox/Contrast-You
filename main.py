import os

import torch
from deepclustering2.configparser import ConfigManger
from deepclustering2.loss import KL_div

from contrastyou import CONFIG_PATH
from contrastyou.utils import fix_all_seed
from semi_seg.arch import UNet
from semi_seg.data import get_data_loaders, create_val_loader
from semi_seg.hooks import create_sp_infonce_hooks, create_discrete_mi_consistency_hook, feature_until_from_hooks
from semi_seg.trainers.new_pretrain import SemiTrainer, PretrainTrainer
from semi_seg.trainers.new_trainer import FineTuneTrainer

trainer_zoo = {"semi": SemiTrainer,
               "ft": FineTuneTrainer,
               "pretrain": PretrainTrainer}

fix_all_seed(1)

with ConfigManger(
    base_path=os.path.join(CONFIG_PATH, "base.yaml"),
    optional_paths=os.path.join(CONFIG_PATH, "specific", "pretrain.yaml"), strict=False,
)(scope="base") as config:
    checkpoint = config["Arch"].pop("checkpoint", None)
    model = UNet(**config["Arch"])
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    labeled_loader, unlabeled_loader, test_loader = get_data_loaders(
        data_params=config["Data"], labeled_loader_params=config["LabeledLoader"],
        unlabeled_loader_params=config["UnlabeledLoader"], pretrain=True, group_test=True, total_freedom=True
    )
    val_loader, test_loader = create_val_loader(test_loader=test_loader)
    trainer_name = config["Trainer"]["name"]
    Trainer = trainer_zoo[trainer_name]

    trainer = Trainer(model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
                      val_loader=val_loader, test_loader=test_loader,
                      criterion=KL_div(), config=config, **config["Trainer"])

    iic_hook = create_discrete_mi_consistency_hook(model=model, feature_names=["Conv5", "Up_conv3", "Up_conv2"],
                                                   mi_weights=[0.1, 0.05, 0.05], dense_paddings=[0, 1],
                                                   consistency_weight=1.0)
    info_hook = create_sp_infonce_hooks(model=model, feature_names=["Conv5", ], weights=0.1,
                                        contrast_ons=["partition", ], data_name="acdc", begin_values=1e6,
                                        end_values=1e6, mode="soft", max_epoch=10, correct_grad=True)

    trainer.register_hooks(iic_hook)

    trainer.forward_until = feature_until_from_hooks(iic_hook, )

    trainer.init()
    trainer.start_training()
