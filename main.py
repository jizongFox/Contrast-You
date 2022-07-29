import os
import typing as t
from contextlib import nullcontext
from pathlib import Path

from loguru import logger

from contrastyou import CONFIG_PATH, git_hash, OPT_PATH, on_cc
from contrastyou.arch import get_arch
from contrastyou.configure.omega_parser import OmegaParser
from contrastyou.losses.kl import KL_div
from contrastyou.trainer import create_save_dir
from contrastyou.utils import fix_all_seed_within_context, adding_writable_sink, extract_model_state_dict
from hook_creator import create_hook_from_config
from semi_seg.data.creator import get_data
from semi_seg.hooks import feature_until_from_hooks
from semi_seg.trainers import trainer_zoo, SemiTrainer
from utils import find_checkpoint


@logger.catch(reraise=True)
def main():
    manager = OmegaParser(os.path.join(CONFIG_PATH, "base.yaml"))
    with manager(scope="base") as config:
        # this handles input save dir with relative and absolute paths

        with create_save_dir(SemiTrainer, config["Trainer"]["save_dir"]) as absolute_save_dir:
            if os.path.exists(absolute_save_dir):
                logger.warning(f"{absolute_save_dir} exists, may overwrite the folder")
        adding_writable_sink(absolute_save_dir)
        logger.info("configuration:\n" + str(manager.summary()))
        with OmegaParser.modifiable_cxm(config, True):
            config.update({"GITHASH": git_hash})

        seed = config.get("RandomSeed", 10)
        logger.info(f"using seed = {seed}, saved at \"{absolute_save_dir}\"")
        with fix_all_seed_within_context(seed):
            worker(config, absolute_save_dir, seed)


def worker(config, absolute_save_dir, seed):
    # load data setting
    data_name = config.Data.name
    data_opt = OmegaParser.load_yaml(Path(OPT_PATH) / f"{data_name}.yaml")
    with OmegaParser.modifiable_cxm(config, True):
        config.OPT = data_opt
        model_checkpoint = config["Arch"].pop("checkpoint", None)

    with fix_all_seed_within_context(seed):
        model = get_arch(input_dim=data_opt.input_dim, num_classes=data_opt.num_classes, **config["Arch"])
    if model_checkpoint:
        logger.info(f"loading model checkpoint from  {model_checkpoint}")
        try:
            model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=True)
            logger.info(f"successfully loaded model checkpoint from  {model_checkpoint}")
        except RuntimeError as e:
            # shape mismatch for network.
            logger.warning(e)

    trainer_name = config["Trainer"]["name"]
    assert trainer_name in trainer_zoo, (trainer_name, trainer_zoo.keys())
    is_pretrain = ("pretrain" in trainer_name)
    order_num = config["Data"]["order_num"]
    labeled_loader, unlabeled_loader, val_loader, test_loader = get_data(
        data_params=config["Data"], labeled_loader_params=config["LabeledLoader"],
        unlabeled_loader_params=config["UnlabeledLoader"], pretrain=is_pretrain, total_freedom=False,
        order_num=order_num
    )
    OmegaParser.set_modifiable(config, True)
    Trainer = trainer_zoo[trainer_name]

    trainer = Trainer(model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
                      val_loader=val_loader, test_loader=test_loader, criterion=KL_div(), config=config,
                      save_dir=absolute_save_dir,
                      **{k: v for k, v in config["Trainer"].items() if k not in ["save_dir", "name"]})

    # find the last.pth from the save folder.
    if on_cc():
        checkpoint: t.Optional[str] = find_checkpoint(trainer.absolute_save_dir)
    else:
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
            until = feature_until_from_hooks(*hooks, model=model, )
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
        trainer.start_training()
        return trainer.inference(checkpoint_path=checkpoint)


if __name__ == '__main__':
    import torch

    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = True  # noqa
    main()
