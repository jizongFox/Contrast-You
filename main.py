import os

from deepclustering2.configparser import ConfigManger
from deepclustering2.loss import KL_div

from contrastyou import CONFIG_PATH
from semi_seg.arch import UNet
from semi_seg.data import get_data_loaders, create_val_loader
from semi_seg.hooks.creator import create_iic_hook
from semi_seg.trainers.new_experiment import SemiTrainer

with ConfigManger(base_path=os.path.join(CONFIG_PATH, "base.yaml"))(scope="base") as config:
    model = UNet(num_classes=4, input_dim=1)


    def get_data_params():
        return {"name": "acdc",
                "labeled_data_ratio": 0.05}


    def get_loader_params():
        return {"shuffle": True, "num_workers": 8, "batch_size": 5}


    labeled_loader, unlabeled_loader, test_loader = get_data_loaders(
        data_params=config["Data"], labeled_loader_params=config["LabeledLoader"],
        unlabeled_loader_params=config["UnlabeledLoader"], pretrain=False, group_test=True, total_freedom=False
    )
    val_loader, test_loader = create_val_loader(test_loader=test_loader)
    trainer = SemiTrainer(model=model, labeled_loader=labeled_loader,
                          unlabeled_loader=unlabeled_loader, val_loader=val_loader, test_loader=test_loader,
                          criterion=KL_div(), config=config, **config["Trainer"])
    iic_hook = create_iic_hook(["Conv5", "Up_conv3", "Up_conv2"], [1, 0.5, 0.5], 10, model=model)
    trainer.register_hook(iic_hook)

    trainer.init()

    trainer.start_training()
