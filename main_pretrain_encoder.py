import os
from copy import deepcopy as dcopy

import numpy  # noqa
from loguru import logger

from contrastyou import CONFIG_PATH, git_hash
from contrastyou.arch import UNet
from contrastyou.configure import ConfigManger
from contrastyou.losses.kl import KL_div
from contrastyou.trainer import create_save_dir
from contrastyou.utils import fix_all_seed_within_context, config_logger, extract_model_state_dict
from hook_creator import create_hook_from_config
from semi_seg import ratio_zoo
from semi_seg.data.creator import get_data
from semi_seg.hooks import feature_until_from_hooks
from semi_seg.trainers.pretrain import PretrainEncoderTrainer
from utils import separate_pretrain_finetune_configs, logging_configs, find_checkpoint
from val import val


def main():
    config_manager = ConfigManger(
        base_path=os.path.join(CONFIG_PATH, "base.yaml"),
        optional_paths=os.path.join(CONFIG_PATH, "pretrain.yaml"), strict=False, verbose=False
    )
    pretrain_config, base_config = separate_pretrain_finetune_configs(config_manager=config_manager)

    with config_manager(scope="base") as config:
        absolute_save_dir = create_save_dir(PretrainEncoderTrainer, config["Trainer"]["save_dir"])
        config_logger(absolute_save_dir)
        logging_configs(config_manager, logger)

        pretrain_config.update({"GITHASH": git_hash})
        base_config.update({"GITHASH": git_hash})

        seed = config.get("RandomSeed", 10)

        data_name = config["Data"]["name"]
        with fix_all_seed_within_context(seed):
            model = worker(pretrain_config, absolute_save_dir, seed)

        val(model=model, save_dir=absolute_save_dir, base_config=base_config, seed=seed,
            labeled_ratios=ratio_zoo[data_name])


def worker(config, absolute_save_dir, seed, ):
    config = dcopy(config)
    model_checkpoint = config["Arch"].pop("checkpoint", None)
    with fix_all_seed_within_context(seed):
        model = UNet(**config["Arch"])
    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=False)

    labeled_loader, unlabeled_loader, val_loader, test_loader = get_data(
        data_params=config["Data"], labeled_loader_params=config["LabeledLoader"],
        unlabeled_loader_params=config["UnlabeledLoader"], pretrain=True, total_freedom=True)

    trainer = PretrainEncoderTrainer(model=model, labeled_loader=labeled_loader,
                                     unlabeled_loader=unlabeled_loader,
                                     val_loader=val_loader, test_loader=test_loader,
                                     criterion=KL_div(), config=config,
                                     save_dir=os.path.join(absolute_save_dir, "pre"),
                                     **{k: v for k, v in config["Trainer"].items() if k != "save_dir"})

    checkpoint = find_checkpoint(trainer.absolute_save_dir)

    with fix_all_seed_within_context(seed):
        hooks = create_hook_from_config(model, config, is_pretrain=True, trainer=trainer)
        assert len(hooks) > 0, "empty hooks"

    trainer.register_hook(*hooks)
    until = feature_until_from_hooks(*hooks)
    assert until == "Conv5"
    trainer.forward_until = until

    with model.set_grad(False, start=until, include_start=False):
        trainer.init()
        if checkpoint:
            trainer.resume_from_path(checkpoint)
        trainer.start_training()

    return model


if __name__ == '__main__':
    import torch

    with logger.catch(reraise=True):
        torch.set_deterministic(True)
        # torch.backends.cudnn.benchmark = True  # noqa
        main()
