import os

from deepclustering2.configparser import ConfigManger
from deepclustering2.loss import KL_div

from contrastyou import CONFIG_PATH
from contrastyou.utils import fix_all_seed
from semi_seg.arch import UNet
from semi_seg.data import get_data_loaders, create_val_loader
from semi_seg.hooks.creator import create_infonce_hook, create_iic_hook
from semi_seg.hooks.entmin import EntropyMinTrainerHook
from semi_seg.trainers.new_pretrain import SemiTrainer

# set_deterministic(True)
fix_all_seed(1)

with ConfigManger(base_path=os.path.join(CONFIG_PATH, "base.yaml"),
                  optional_paths=os.path.join(CONFIG_PATH, "specific", "pretrain.yaml")
                  )(scope="base") as config:
    checkpoint = config["Arch"].pop("checkpoint", None)
    model = UNet(**config["Arch"])

    labeled_loader, unlabeled_loader, test_loader = get_data_loaders(
        data_params=config["Data"], labeled_loader_params=config["LabeledLoader"],
        unlabeled_loader_params=config["UnlabeledLoader"], pretrain=False, group_test=True, total_freedom=False
    )
    val_loader, test_loader = create_val_loader(test_loader=test_loader)
    trainer = SemiTrainer(model=model, labeled_loader=labeled_loader,
                          unlabeled_loader=unlabeled_loader, val_loader=val_loader, test_loader=test_loader,
                          criterion=KL_div(), config=config, inference_until="Conv5", **config["Trainer"])
    iic_hook = create_iic_hook(["Conv5", "Up_conv3", "Up_conv2"], [0.1, 0.05, 0.05], 5, model=model)
    trainer.register_hook(iic_hook)
    # infonce_hook = create_infonce_hook(["Conv5"], weights=[1],
    #                                    contrast_ons=["partition", ], mode="soft",
    #                                    begin_values=[1e6], end_values=[1e6],
    #                                    max_epoch=config["Trainer"]["max_epoch"], model=model,
    #                                    data_name=config["Data"]["name"])
    # trainer.register_hook(infonce_hook)
    # trainer.register_hook(EntropyMinTrainerHook(name="ent", weight=0.1))
    # with model.set_grad(enable=False, start="Conv5", include_start=False, ):
    trainer.init()
    trainer.start_training()
