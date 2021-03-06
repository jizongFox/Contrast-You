import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np  # noqa
from contrastyou import PROJECT_PATH
from contrastyou.arch import UNet
from contrastyou.arch.unet import arch_order
from contrastyou.helper import extract_model_state_dict
from deepclustering2.configparser import ConfigManger
from deepclustering2.loss import KL_div
from deepclustering2.utils import set_benchmark, gethash, yaml_load, fix_all_seed
from loguru import logger
from semi_seg.dsutils import get_dataloaders
from semi_seg.trainers import pre_trainer_zoos, base_trainer_zoos, FineTuneTrainer

cur_githash = gethash(__file__)  # noqa

trainer_zoos = {**base_trainer_zoos, **pre_trainer_zoos}


def main():
    config_manager = ConfigManger(base_path=Path(PROJECT_PATH) / "config/base.yaml")
    with config_manager(scope="base") as config:
        port = random.randint(10000, 60000)
        parsed_config = config_manager.parsed_config
        if "Optim" in parsed_config and "lr" in parsed_config["Optim"]:
            raise RuntimeError("`Optim.lr` should not be provided in this interface, "
                               "Provide `Optim.pre_lr` and `Optim.ft_lr` instead ")

        main_worker(0, 1, config, config_manager, port)


@logger.catch(reraise=True)
def main_worker(rank, ngpus_per_node, config, config_manager, port):  # noqa
    base_save_dir = str(config["Trainer"]["save_dir"])
    logger.add(os.path.join("runs", base_save_dir, "loguru.log"), level="TRACE", diagnose=True)

    fix_all_seed(config.get("RandomSeed", 1))

    labeled_loader, unlabeled_loader, val_loader = get_dataloaders(config)

    config_arch = deepcopy(config["Arch"])
    model_checkpoint = config_arch.pop("checkpoint", None)
    model = UNet(**config_arch)
    logger.info(f"Initializing {model.__class__.__name__}")
    if model_checkpoint:
        logger.info(f"loading checkpoint from  {model_checkpoint}")
        model.load_state_dict(extract_model_state_dict(model_checkpoint), strict=False)

    trainer_name = config["Trainer"]["name"]
    assert trainer_name in ("infoncepretrain", "experimentpretrain", "experimentmixuppretrain"), trainer_name  # noqa

    Trainer = trainer_zoos[trainer_name]

    is_pretrain: bool = trainer_name in pre_trainer_zoos

    pre_lr = config["Optim"].pop("pre_lr", None)
    ft_lr = config["Optim"].pop("ft_lr", None)

    if is_pretrain:
        if pre_lr is not None:
            config["Optim"]["lr"] = float(pre_lr)

    trainer = Trainer(
        model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
        val_loader=val_loader, sup_criterion=KL_div(verbose=False),
        configuration={**config, **{"GITHASH": cur_githash}},
        save_dir=os.path.join(base_save_dir, "pre") if is_pretrain else base_save_dir,
        **{k: v for k, v in config["Trainer"].items() if k != "save_dir" and k != "name"}
    )
    trainer.init()
    trainer_checkpoint = config.get("trainer_checkpoint", None)
    if trainer_checkpoint:
        trainer.load_state_dict_from_path(trainer_checkpoint, strict=True)

    if not is_pretrain:
        trainer.start_training()
    else:
        if "FeatureExtractor" not in trainer._config:  # noqa
            raise RuntimeError("FeatureExtractor should be in trainer config")
        from_, util_ = \
            config["Trainer"]["grad_from"] or "Conv1", \
            config["Trainer"]["grad_util"] or \
            sorted(trainer._config["FeatureExtractor"]["feature_names"], key=lambda x: arch_order(x))[-1]  # noqa
        with \
            trainer.enable_grad(from_=from_, util_=util_), \
            trainer.enable_bn(from_=from_, util_=util_):
            trainer.start_training()

        for labeled_ratio in (0.01, 0.03, 0.05, 1.0):
            model.load_state_dict(extract_model_state_dict(
                os.path.join(trainer._save_dir, "last.pth")),  # noqa
                strict=True
            )
            # get the base configuration
            base_config = config_manager.base_config
            # in the cmd, one can change `Data` and `Trainer`
            base_config["Data"]["name"] = config["Data"]["name"]
            base_config["Data"]["labeled_data_ratio"] = labeled_ratio
            base_config["Data"]["unlabeled_data_ratio"] = 1 - labeled_ratio
            # modifying the optimizer options
            use_different_lr = config["Use_diff_lr"]
            if use_different_lr:
                advanced_optim_config = yaml_load("../config/specific/optim.yaml", verbose=False)
                base_config.update(advanced_optim_config)
                base_config["OptimizerSupplementary"]["name"] = config["Optim"]["name"]
                base_config["OptimizerSupplementary"]["group"]["feature_names"] = [
                    f for f in model.component_names if
                    model.component_names.index(from_) <= model.component_names.index(f) <= model.component_names.index(
                        util_)
                ]
            if ft_lr is not None:
                config["Optim"]["lr"] = float(ft_lr)
                # update trainer
            base_config["Trainer"].update(config["Trainer"])

            labeled_loader, unlabeled_loader, val_loader = get_dataloaders(base_config)

            finetune_trainer = FineTuneTrainer(
                model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
                val_loader=val_loader, sup_criterion=KL_div(verbose=False),
                configuration={**base_config, **{"GITHASH": cur_githash}},
                save_dir=os.path.join(base_save_dir, "tra", f"ratio_{str(labeled_ratio)}"),
                **{k: v for k, v in base_config["Trainer"].items() if k != "save_dir"}
            )
            finetune_trainer.init()
            finetune_trainer.start_training()
            # finetune_trainer.inference()


if __name__ == '__main__':
    main()
