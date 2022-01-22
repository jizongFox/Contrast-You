# the adversarial training use trainer and epochers directly, without using the hook, since it consists of multiple
# gradient steps.
import os
from pathlib import Path

import torch
from easydict import EasyDict as edict
from loguru import logger

from contrastyou import CONFIG_PATH, OPT_PATH, git_hash
from contrastyou.arch import UNet
from contrastyou.configure import ConfigManager, yaml_load
from contrastyou.losses.kl import KL_div
from contrastyou.trainer import create_save_dir
from contrastyou.utils import fix_all_seed_within_context, adding_writable_sink, extract_model_state_dict, get_dataset
from semi_seg.data.creator import get_data
from semi_seg.trainers.trainer import AdversarialTrainer
from utils import logging_configs


def main():
    manager = ConfigManager(
        os.path.join(CONFIG_PATH, "base.yaml"), strict=True
    )
    with manager(scope="base") as config:
        # this handles input save dir with relative and absolute paths
        absolute_save_dir = create_save_dir(AdversarialTrainer, config["Trainer"]["save_dir"])
        if os.path.exists(absolute_save_dir):
            logger.warning(f"{absolute_save_dir} exists, may overwrite the folder")
        adding_writable_sink(absolute_save_dir)
        logging_configs(manager, logger)

        config.update({"GITHASH": git_hash})

        seed = config.get("RandomSeed", 10)
        logger.info(f"using seed = {seed}, saved at \"{absolute_save_dir}\"")
        with fix_all_seed_within_context(seed):
            worker(config, absolute_save_dir, seed)


def worker(config, absolute_save_dir, seed, ):
    data_name = config.Data.name
    data_opt = yaml_load(Path(OPT_PATH) / (data_name + ".yaml"))
    data_opt = edict(data_opt)
    config.OPT = data_opt

    model_checkpoint = config["Arch"].pop("checkpoint", None)
    with fix_all_seed_within_context(seed):
        model = UNet(input_dim=data_opt.input_dim, num_classes=data_opt.num_classes, **config["Arch"])
    if model_checkpoint:
        logger.info(f"loading model checkpoint from  {model_checkpoint}")
        try:
            model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=True)
            logger.info(f"successfully loaded model checkpoint from  {model_checkpoint}")
        except RuntimeError as e:
            # shape mismatch for network.
            logger.warning(e)

    labeled_loader, unlabeled_loader, val_loader, test_loader = get_data(
        data_params=config["Data"], labeled_loader_params=config["LabeledLoader"],
        unlabeled_loader_params=config["UnlabeledLoader"], pretrain=False, total_freedom=True)
    unlabeled_data = get_dataset(unlabeled_loader)
    if len(unlabeled_data) == 0:
        logger.warning(f"Detected full supervised training, exiting with success.")
        exit(0)

    checkpoint = config.get("trainer_checkpoint")

    trainer = AdversarialTrainer(model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
                                 val_loader=val_loader, test_loader=test_loader,
                                 criterion=KL_div(), config=config,
                                 save_dir=absolute_save_dir, dis_consider_image=True,
                                 **{k: v for k, v in config["Trainer"].items() if k != "save_dir" and k != "name"})

    trainer.init()
    if checkpoint:
        trainer.resume_from_path(checkpoint)
    trainer.start_training()


if __name__ == '__main__':
    # set_deterministic(True)
    torch.backends.cudnn.benchmark = True  # noqa
    main()
