import os
import typing as t
from pathlib import Path

from easydict import EasyDict as edict
from loguru import logger

from contrastyou import CONFIG_PATH, git_hash, OPT_PATH, on_cc
from contrastyou.arch import UNet
from contrastyou.configure import yaml_load, ConfigManager
from contrastyou.losses.kl import KL_div
from contrastyou.trainer import create_save_dir
from contrastyou.utils import fix_all_seed_within_context, extract_model_state_dict
from hook_creator import create_hook_from_config
from semi_seg.data.creator import get_data
from semi_seg.hooks import feature_until_from_hooks
from semi_seg.trainers.pretrain import PretrainDecoderTrainer
from semi_seg.trainers.trainer import SemiTrainer
from utils import logging_configs, find_checkpoint  # noqa


@logger.catch(reraise=True)
def main():
    manager = ConfigManager(os.path.join(CONFIG_PATH, "base.yaml"), strict=True, verbose=False)
    with manager(scope="base") as config:
        # this handles input save dir with relative and absolute paths
        absolute_save_dir = create_save_dir(SemiTrainer, config["Trainer"]["save_dir"])
        if os.path.exists(absolute_save_dir):
            logger.warning(f"{absolute_save_dir} exists, may overwrite the folder")

        config.update({"GITHASH": git_hash})

        seed = config.get("RandomSeed", 10)
        logger.info(f"using seed = {seed}, saved at \"{absolute_save_dir}\"")
        with fix_all_seed_within_context(seed):
            worker(config, absolute_save_dir, seed)


def worker(config, absolute_save_dir, seed):
    # load data setting
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

    total_freedom = False
    is_pretrain = False
    order_num = config["Data"]["order_num"]
    labeled_loader, unlabeled_loader, val_loader, test_loader = get_data(
        data_params=config["Data"], labeled_loader_params=config["LabeledLoader"],
        unlabeled_loader_params=config["UnlabeledLoader"], pretrain=is_pretrain, total_freedom=total_freedom,
        order_num=order_num
    )

    Trainer: 'Trainer' = PretrainDecoderTrainer

    trainer = Trainer(
        model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
        val_loader=val_loader, test_loader=test_loader, criterion=KL_div(), config=config, save_dir=absolute_save_dir,
        **{k: v for k, v in config["Trainer"].items() if k != "save_dir" and k != "name"}
    )
    # find the last.pth from the save folder.
    if on_cc():
        checkpoint: t.Optional[str] = find_checkpoint(trainer.absolute_save_dir)
    else:
        checkpoint: t.Optional[str] = config.trainer_checkpoint

    with fix_all_seed_within_context(seed):
        hooks = create_hook_from_config(model, config, is_pretrain=False, trainer=trainer)
        assert len(hooks) > 0

    hook_registration = trainer.register_hook
    with hook_registration(*hooks):
        until = feature_until_from_hooks(*hooks)
        trainer.forward_until = until
        with model.switch_grad(False, start=until, include_start=False):
            trainer.init()
            if checkpoint:
                trainer.resume_from_path(checkpoint)
            trainer.start_training()
            os.environ["contrast_save_flag"] = "true"

            trainer.inference()


if __name__ == '__main__':
    import torch

    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = True  # noqa
    main()
