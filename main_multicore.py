import os
import typing as t
from pathlib import Path

from easydict import EasyDict as edict
from loguru import logger

from contrastyou import CONFIG_PATH, git_hash, OPT_PATH
from contrastyou.arch import UNet
from contrastyou.configure import ConfigManager
from contrastyou.configure.yaml_parser import yaml_load
from contrastyou.losses.multicore_loss import MultiCoreKL
from contrastyou.trainer import create_save_dir
from contrastyou.utils import fix_all_seed_within_context, adding_writable_sink, extract_model_state_dict
from hook_creator import create_hook_from_config
from semi_seg.data.creator import get_data
from semi_seg.trainers.features import MulticoreTrainer
from utils import logging_configs, find_checkpoint, grouper


@logger.catch(reraise=True)
def main():
    manager = ConfigManager(os.path.join(CONFIG_PATH, "base.yaml"), strict=True, verbose=False)

    with manager(scope="base") as config:
        # this handles input save dir with relative and absolute paths
        absolute_save_dir = create_save_dir(MulticoreTrainer, config["Trainer"]["save_dir"])
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
    data_opt = edict(yaml_load(Path(OPT_PATH) / (data_name + ".yaml")))
    config.OPT = data_opt

    model_checkpoint = config["Arch"].pop("checkpoint", None)

    with fix_all_seed_within_context(seed):
        config["Arch"].pop("true_num_classes", None)
        true_num_classes = data_opt["num_classes"]
        multiplier = config["MulticoreParameters"]["multiplier"]
        model = UNet(**config["Arch"], input_dim=data_opt["input_dim"], num_classes=multiplier * true_num_classes)
        config["Arch"]["true_num_classes"] = true_num_classes
        criterion_name = config["MulticoreParameters"]["name"]
        if criterion_name == "naive":
            sup_criterion = MultiCoreKL(groups=list(grouper(range(true_num_classes * multiplier), true_num_classes)))
        else:
            raise RuntimeError(criterion_name)
            # sup_criterion = StricterAdaptiveOverSegmentedLossWithMI(
            #     input_num_classes=true_num_classes * multiplier,
            #     output_num_classes=true_num_classes,
            #     device=config.Trainer.device,
            #     mi_weight=config["MulticoreParameters"]["mi_weight"]
            # )

    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=True)

    total_freedom = False

    labeled_loader, unlabeled_loader, val_loader, test_loader = get_data(
        data_params=config["Data"], labeled_loader_params=config["LabeledLoader"],
        unlabeled_loader_params=config["UnlabeledLoader"], pretrain=False, total_freedom=total_freedom)

    trainer = MulticoreTrainer(
        model=model, labeled_loader=labeled_loader, unlabeled_loader=unlabeled_loader,
        val_loader=val_loader, test_loader=test_loader, criterion=sup_criterion, config=config,
        save_dir=absolute_save_dir,
        **{k: v for k, v in config["Trainer"].items() if k != "save_dir" and k != "name"}
    )
    # find the last.pth from the save folder.
    checkpoint: t.Optional[str] = find_checkpoint(trainer.absolute_save_dir)

    with fix_all_seed_within_context(seed):
        hooks = create_hook_from_config(model, config, is_pretrain=False, trainer=trainer)
    hook_registration = trainer.register_hook
    with hook_registration(*hooks):
        trainer.init()
        if checkpoint:
            trainer.resume_from_path(checkpoint)
        return trainer.start_training()


if __name__ == '__main__':
    import torch

    torch.backends.cudnn.benchmark = True
    main()
