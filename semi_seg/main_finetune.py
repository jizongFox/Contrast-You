import os

from scipy.sparse import issparse  # noqa

_ = issparse  # noqa
from contrastyou.helper import extract_model_state_dict
from deepclustering2.loss import KL_div
import random
from pathlib import Path
from contrastyou import PROJECT_PATH
from contrastyou.arch import UNet
from deepclustering2.configparser import ConfigManger
from deepclustering2.utils import gethash
from deepclustering2.utils import set_benchmark
from semi_seg.trainers import pre_trainer_zoos, base_trainer_zoos, FineTuneTrainer
from semi_seg.dsutils import get_dataloaders
from loguru import logger
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")

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

    set_benchmark(config.get("RandomSeed", 1))

    config_arch = deepcopy(config["Arch"])
    model_checkpoint = config_arch.pop("checkpoint", None)
    model = UNet(**config_arch)
    logger.info(f"Initializing {model.__class__.__name__}")
    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=False)

    trainer_name = config["Trainer"].pop("name")
    assert trainer_name in ("finetune",), trainer_name
    base_model_checkpoint = model.state_dict()

    for labeled_ratio in (0.01, 0.02, 0.03, 0.04, 0.05, 1.0):
        model.load_state_dict(base_model_checkpoint)

        config["Data"]["labeled_data_ratio"] = labeled_ratio
        config["Data"]["unlabeled_data_ratio"] = 1 - labeled_ratio

        labeled_loader, unlabeled_loader, val_loader = get_dataloaders(config)

        finetune_trainer = FineTuneTrainer(
            model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
            val_loader=val_loader, sup_criterion=KL_div(verbose=False),
            configuration={**config, **{"GITHASH": cur_githash}},
            save_dir=os.path.join(base_save_dir, "tra",
                                  f"ratio_{str(labeled_ratio)}"),
            **{k: v for k, v in config["Trainer"].items() if k != "save_dir"}
        )
        finetune_trainer.init()
        finetune_trainer.start_training()
        finetune_trainer.inference()


if __name__ == '__main__':
    main()
