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
from semi_seg.trainers import base_trainer_zoos
from semi_seg.trainers import pre_trainer_zoos
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
        save_dir = config["Trainer"]["save_dir"]
        logger.add(os.path.join(save_dir, "loguru.log"))

        port = random.randint(10000, 60000)
        use_distributed_training = config.get("DistributedTrain")

        if use_distributed_training is True:
            """ngpus_per_node = torch.cuda.device_count()
            print(colored(f"Found {ngpus_per_node} per node", "red"))
            mp.spawn(main_worker, nprocs=ngpus_per_node,  # noqa
                     args=(ngpus_per_node, config, port))
            """
            raise RuntimeError("DDP training not supported")
        else:
            main_worker(0, 1, config, port)


@logger.catch(reraise=True)
def main_worker(rank, ngpus_per_node, config, port):  # noqa
    set_benchmark(config.get("RandomSeed", 1))

    labeled_loader, unlabeled_loader, val_loader = get_dataloaders(config)

    config_arch = deepcopy(config["Arch"])
    model_checkpoint = config_arch.pop("checkpoint", None)
    model = UNet(**config_arch)
    logger.info(f"Initializing {model.__class__.__name__}")
    if model_checkpoint:
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=False)
        logger.info(f"loading checkpoint from  {model_checkpoint}")

    trainer_name = config["Trainer"].pop("name")
    Trainer = trainer_zoos[trainer_name]

    trainer = Trainer(
        model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
        val_loader=val_loader, sup_criterion=KL_div(verbose=False),
        configuration={**config, **{"GITHASH": cur_githash}},
        **config["Trainer"]
    )

    trainer.init()
    trainer_checkpoint = config.get("trainer_checkpoint", None)
    if trainer_checkpoint:
        trainer.load_state_dict_from_path(trainer_checkpoint, strict=True)

    trainer.start_training()
    if "pretrain" not in trainer_name:
        trainer.inference()


if __name__ == '__main__':
    main()
