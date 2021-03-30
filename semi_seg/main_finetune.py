import os
import random
from copy import deepcopy
from pathlib import Path

import numpy  # noqa
from deepclustering2.configparser import ConfigManger
from deepclustering2.loss import KL_div
from deepclustering2.utils import gethash, fix_all_seed
from loguru import logger

from contrastyou import PROJECT_PATH
from contrastyou.arch import UNet
from contrastyou.helper import extract_model_state_dict
from semi_seg.dsutils import get_dataloaders
from semi_seg.trainers import pre_trainer_zoos, base_trainer_zoos, DirectTrainer

cur_githash = gethash(__file__)  # noqa

trainer_zoos = {**base_trainer_zoos, **pre_trainer_zoos}


def main():
    config_manager = ConfigManger(base_path=Path(PROJECT_PATH) / "config/base.yaml")
    with config_manager(scope="base") as config:
        port = random.randint(10000, 60000)
        main_worker(0, 1, config, config_manager, port)


@logger.catch(reraise=True)
def main_worker(rank, ngpus_per_node, config, config_manager, port):  # noqa
    base_save_dir = str(config["Trainer"]["save_dir"])
    logger.add(os.path.join("runs", base_save_dir, "loguru.log"), level="TRACE", diagnose=True, )

    fix_all_seed(config.get("RandomSeed", 1))

    config_arch = deepcopy(config["Arch"])
    model_checkpoint = config_arch.pop("checkpoint", None)
    model = UNet(**config_arch)
    logger.info(f"Initializing {model.__class__.__name__}")
    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=False)

    trainer_name = config["Trainer"].pop("name")
    assert trainer_name in ("finetune", "directtrain"), trainer_name
    base_model_checkpoint = deepcopy(model.state_dict())

    if config["Data"]["name"] == "acdc":
        ratios = (0.01, 0.015, 0.025, 1.0)
    else:
        ratios = (0.1, 0.14, 0.18, 0.2,)

    for labeled_ratio in ratios:
        model.load_state_dict(base_model_checkpoint)

        config["Data"]["labeled_data_ratio"] = labeled_ratio
        config["Data"]["unlabeled_data_ratio"] = 1 - labeled_ratio

        labeled_loader, unlabeled_loader, val_loader = get_dataloaders(config)

        finetune_trainer = DirectTrainer(
            model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
            val_loader=val_loader, sup_criterion=KL_div(verbose=False),
            configuration={**config, **{"GITHASH": cur_githash}},
            save_dir=os.path.join(base_save_dir, "tra",
                                  f"ratio_{str(labeled_ratio)}"),
            **{k: v for k, v in config["Trainer"].items() if k != "save_dir"}
        )
        finetune_trainer.init()
        finetune_trainer.start_training()
        # finetune_trainer.inference()


if __name__ == '__main__':
    main()
