import os
from contextlib import nullcontext

import numpy  # noqa
from loguru import logger

from contrastyou import CONFIG_PATH, git_hash
from contrastyou.arch import UNet
from contrastyou.configure import ConfigManger
from contrastyou.losses.kl import KL_div
from contrastyou.trainer import create_save_dir
from contrastyou.utils import fix_all_seed_within_context, config_logger, extract_model_state_dict
from hook_creator import create_hook_from_config
from semi_seg.data.creator import get_data
from semi_seg.hooks import feature_until_from_hooks
from semi_seg.trainers.pretrain import PretrainEncoderTrainer
from semi_seg.trainers.trainer import SemiTrainer, FineTuneTrainer, MixUpTrainer, MTTrainer
from utils import logging_configs, find_checkpoint

trainer_zoo = {"semi": SemiTrainer,
               "ft": FineTuneTrainer,
               "pretrain": PretrainEncoderTrainer,
               "mt": MTTrainer,
               "mixup": MixUpTrainer}


def main():
    manager = ConfigManger(base_path=os.path.join(CONFIG_PATH, "base.yaml"), strict=True, verbose=False)
    with manager(scope="base") as config:
        # this handles input save dir with relative and absolute paths
        absolute_save_dir = create_save_dir(SemiTrainer, config["Trainer"]["save_dir"])
        config_logger(absolute_save_dir)
        logging_configs(manager, logger)

        config.update({"GITHASH": git_hash})

        seed = config.get("RandomSeed", 10)
        logger.info(f"using seed = {seed}, saved at \"{absolute_save_dir}\"")
        with fix_all_seed_within_context(seed):
            worker(config, absolute_save_dir, seed)


def worker(config, absolute_save_dir, seed):
    model_checkpoint = config["Arch"].pop("checkpoint", None)
    with fix_all_seed_within_context(seed):
        model = UNet(**config["Arch"])
    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=True)

    trainer_name = config["Trainer"]["name"]
    is_pretrain = trainer_name == "pretrain"
    total_freedom = True if is_pretrain or trainer_name == "mixup" else False

    labeled_loader, unlabeled_loader, val_loader, test_loader = get_data(
        data_params=config["Data"], labeled_loader_params=config["LabeledLoader"],
        unlabeled_loader_params=config["UnlabeledLoader"], pretrain=is_pretrain, total_freedom=total_freedom)

    Trainer = trainer_zoo[trainer_name]

    trainer = Trainer(
        model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
        val_loader=val_loader, test_loader=test_loader, criterion=KL_div(), config=config, save_dir=absolute_save_dir,
        **{k: v for k, v in config["Trainer"].items() if k != "save_dir" and k != "name"}
    )
    # find the last.pth from the save folder.
    checkpoint = find_checkpoint(trainer.absolute_save_dir)

    if trainer_name != "ft":
        with fix_all_seed_within_context(seed):
            hooks = create_hook_from_config(model, config, is_pretrain=is_pretrain, trainer=trainer)
            assert len(hooks) > 0, f"You should provide `Hook` configuration for `{trainer_name}` Trainer"

    hook_registration = trainer.register_hook if trainer_name != "ft" else nullcontext

    with hook_registration(*hooks):
        if trainer_name == "pretrain":
            until = feature_until_from_hooks(*hooks)
            trainer.forward_until = until
            with model.set_grad(False, start=until, include_start=False):
                trainer.init()
                if checkpoint:
                    trainer.resume_from_path(checkpoint)
                return trainer.start_training()
        # semi + ft
        trainer.init()
        if checkpoint:
            trainer.resume_from_path(checkpoint)
        return trainer.start_training()


if __name__ == '__main__':
    import torch

    with logger.catch(reraise=True):
        torch.set_deterministic(True)
        # torch.backends.cudnn.benchmark = True  # noqa
        main()
