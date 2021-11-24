import os
import typing as t
from contextlib import nullcontext
from pathlib import Path

from easydict import EasyDict as edict
from loguru import logger

from contrastyou import CONFIG_PATH, git_hash, OPT_PATH
from contrastyou.arch import UNet, UNetFeatureMapEnum
from contrastyou.configure import yaml_load, ConfigManager
from contrastyou.losses.kl import KL_div
from contrastyou.trainer import create_save_dir
from contrastyou.utils import fix_all_seed_within_context, adding_writable_sink, extract_model_state_dict
from hook_creator import create_hook_from_config
from semi_seg.data.creator import get_data
from semi_seg.hooks import feature_until_from_hooks
from semi_seg.trainers.pretrain import PretrainEncoderTrainer, PretrainDecoderTrainer
from semi_seg.trainers.trainer import SemiTrainer, FineTuneTrainer, MixUpTrainer, MTTrainer, DMTTrainer
from utils import logging_configs, randomnize_model

trainer_zoo = {"semi": SemiTrainer,
               "ft": FineTuneTrainer,
               "pretrain": PretrainEncoderTrainer,
               "pretrain_decoder": PretrainDecoderTrainer,
               "mt": MTTrainer,
               "dmt": DMTTrainer,
               "mixup": MixUpTrainer}


@logger.catch(reraise=True)
def main():
    manager = ConfigManager(os.path.join(CONFIG_PATH, "base.yaml"), strict=True, verbose=False)
    with manager(scope="base") as config:
        # this handles input save dir with relative and absolute paths
        absolute_save_dir = create_save_dir(SemiTrainer, config["Trainer"]["save_dir"])
        if os.path.exists(absolute_save_dir):
            logger.warning(f"{absolute_save_dir} exists, may overwrite the folder")
        adding_writable_sink(absolute_save_dir)
        logging_configs(manager, logger)

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

    # this flag is made to randomnize some layers of the decoder, in order to check if it is the optimization that
    # hinders the performance.
    randomnize_checkpoint: bool = config["Arch"].pop("randomnize_checkpoint", False)
    with fix_all_seed_within_context(seed):
        model = UNet(input_dim=data_opt.input_dim, num_classes=data_opt.num_classes, **config["Arch"])
    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=True)
        if randomnize_checkpoint:
            # randomnize some layers if pretrain is used.
            model = randomnize_model(model, start=UNetFeatureMapEnum.Up_conv3, end=UNetFeatureMapEnum.Deconv_1x1)

    trainer_name = config["Trainer"]["name"]
    is_pretrain = ("pretrain" in trainer_name)
    total_freedom = True if is_pretrain or trainer_name == "mixup" else False
    if "CrossCorrelationParameters" in config:
        total_freedom = False
    order_num = config["Data"]["order_num"]
    labeled_loader, unlabeled_loader, val_loader, test_loader = get_data(
        data_params=config["Data"], labeled_loader_params=config["LabeledLoader"],
        unlabeled_loader_params=config["UnlabeledLoader"], pretrain=is_pretrain, total_freedom=total_freedom,
        order_num=order_num
    )

    Trainer: 'Trainer' = trainer_zoo[trainer_name]

    trainer = Trainer(
        model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
        val_loader=val_loader, test_loader=test_loader, criterion=KL_div(), config=config, save_dir=absolute_save_dir,
        **{k: v for k, v in config["Trainer"].items() if k != "save_dir" and k != "name"}
    )
    # find the last.pth from the save folder.
    # checkpoint: t.Optional[str] = find_checkpoint(trainer.absolute_save_dir)
    checkpoint: t.Optional[str] = config.trainer_checkpoint

    if trainer_name not in ("ft", "dmt"):
        with fix_all_seed_within_context(seed):
            hooks = create_hook_from_config(model, config, is_pretrain=is_pretrain, trainer=trainer)
            assert len(hooks) > 0, f"You should provide `Hook` configuration for `{trainer_name}` Trainer"
    else:
        hooks = []
    hook_registration = trainer.register_hook if trainer_name not in ("ft", "dmt") else nullcontext

    with hook_registration(*hooks):
        if is_pretrain:
            until = feature_until_from_hooks(*hooks)
            trainer.forward_until = until
            with model.switch_grad(False, start=until, include_start=False):
                trainer.init()
                if checkpoint:
                    trainer.resume_from_path(checkpoint)
                return trainer.start_training()
        # semi + ft +dmt
        trainer.init()
        if checkpoint:
            trainer.resume_from_path(checkpoint)
        return trainer.start_training()


if __name__ == '__main__':
    import torch

    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = True  # noqa
    main()
