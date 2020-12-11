import warnings

warnings.filterwarnings("ignore")
from scipy.sparse import issparse  # noqa

_ = issparse  # noqa
import os
from deepclustering2.loss import KL_div
import random
from pathlib import Path
from contrastyou import PROJECT_PATH
from contrastyou.arch import UNet
import torch.multiprocessing as mp
from deepclustering2.configparser import ConfigManger
from deepclustering2.utils import gethash, path2Path, yaml_write
import torch
from deepclustering2.utils import set_benchmark, load_yaml, merge_dict
from semi_seg.trainers import trainer_zoos, InfoNCEPretrainTrainer
from semi_seg.innovate import trainer_zoos as trainer_zoos2
from semi_seg.dsutils import get_dataloaders
from deepclustering2.ddp import initialize_ddp_environment, convert2syncBN
from termcolor import colored

cur_githash = gethash(__file__)

trainer_zoos = {**trainer_zoos, **trainer_zoos2}


def main():
    cmanager = ConfigManger(Path(PROJECT_PATH) / "config/semi.yaml")
    config = cmanager.config

    def load_pretrain_dict(cmanager):
        """
        PretrainConfig:
            use_pretrain: bool
            InfoNCEParameters:
                a:b
        """
        try:
            # if load from a checkpoint
            pretrain_config = load_yaml(Path(PROJECT_PATH) / "config/pretrain.yaml", verbose=False)
            if cmanager._default_path is not None:  # noqd
                config_path = path2Path(cmanager._default_path)
                if config_path.is_file():
                    config_path = config_path.parent
                pretrain_config = load_yaml(config_path / "pretrain.yaml", verbose=False)
            # get additional config dict from commandlines
            _received_cmds = cmanager.parsed_config.get("PretrainConfig")
            received_cmds = {"PretrainConfig": _received_cmds} if _received_cmds is not None else None
            pretrain_config = merge_dict(pretrain_config, received_cmds)
        except FileNotFoundError:
            pretrain_config = {"PretrainConfig": {}}
        return pretrain_config

    pretrain_config = load_pretrain_dict(cmanager)

    port = random.randint(10000, 60000)
    use_distributed_training = config.get("DistributedTrain")

    if use_distributed_training is True:
        ngpus_per_node = torch.cuda.device_count()
        print(colored(f"Found {ngpus_per_node} per node", "red"))
        mp.spawn(main_worker, nprocs=ngpus_per_node,  # noqa
                 args=(ngpus_per_node, config, pretrain_config, cmanager, port))
    else:
        main_worker(0, 1, config, pretrain_config, cmanager, port)


def main_worker(rank, ngpus_per_node, config, pretrain_config, cmanager, port):  # noqa
    use_distributed_train = config.get("DistributedTrain")

    if use_distributed_train:
        def set_distributed_training():
            config["Trainer"]["device"] = f"cuda:{rank}"
            config["LabeledData"]["batch_size"] = config["LabeledData"]["batch_size"] // ngpus_per_node
            config["UnlabeledData"]["batch_size"] = config["UnlabeledData"]["batch_size"] // ngpus_per_node
            initialize_ddp_environment(rank, ngpus_per_node, dist_url=f"tcp://localhost:{port}")

        set_distributed_training()

    set_benchmark(config.get("RandomSeed", 1))

    labeled_loader, unlabeled_loader, val_loader = get_dataloaders(config)

    model = UNet(**config["Arch"])
    if use_distributed_train:
        model = convert2syncBN(model)
        model = torch.nn.parallel.DistributedDataParallel(model.to(rank), device_ids=[rank])

    use_pretrain = pretrain_config["PretrainConfig"].get("use_pretrain", False)
    checkpoint = config.get("Checkpoint", None)
    use_only_labeled_data = config["Trainer"].pop("only_labeled_data")
    two_stage_training = config["Trainer"].pop("two_stage_training")
    relative_main_dir = config["Trainer"]["save_dir"]

    ########################## pretraining launch part ################################################
    if use_pretrain:
        # save pretrain config
        pretrainTrainer = InfoNCEPretrainTrainer(
            model=model,
            labeled_loader=iter(labeled_loader),
            unlabeled_loader=iter(unlabeled_loader),
            val_loader=val_loader,
            sup_criterion=KL_div(verbose=False),
            configuration=pretrain_config["PretrainConfig"],
            save_dir=os.path.join(relative_main_dir, "pretrain"),
            **{k: v for k, v in pretrain_config["PretrainConfig"]["Trainer"].items() if k != "save_dir"}
        )
        # manually write config to the save_dir
        absolute_main_dir = relative_main_dir if os.path.isabs(relative_main_dir) else os.path.join(
            pretrainTrainer.RUN_PATH, relative_main_dir)
        yaml_write(pretrain_config, absolute_main_dir, save_name="pretrain.yaml")
        pretrainTrainer.init()
        use_only_labeled_data = True

        if checkpoint is not None:
            pretrainTrainer.load_state_dict_from_path(
                os.path.join(checkpoint, "pretrain"),
                strict=True
            )
        pretrainTrainer.start_training()

    ############################### training launch part ################################################
    trainer_name = config["Trainer"].pop("name")
    Trainer = trainer_zoos[trainer_name]

    trainer = Trainer(
        model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
        val_loader=val_loader, sup_criterion=KL_div(verbose=False),
        configuration={**cmanager.config, **{"GITHASH": cur_githash}},
        save_dir=relative_main_dir if not use_pretrain else os.path.join(relative_main_dir, "train"),
        **{k: v for k, v in config["Trainer"].items() if k != "save_dir"}
    )
    # save config yaml
    absolute_main_dir = relative_main_dir if os.path.isabs(relative_main_dir) else os.path.join(
        trainer.RUN_PATH, relative_main_dir)
    # manually write config to the save_dir
    yaml_write({**cmanager.config, **{"GITHASH": cur_githash}}, absolute_main_dir, save_name="config.yaml")
    trainer.init()

    if checkpoint is not None:
        try:
            trainer.load_state_dict_from_path(os.path.join(checkpoint, "train"), strict=True)
        except Exception:
            trainer.load_state_dict_from_path(checkpoint, strict=True)

    if use_only_labeled_data:
        trainer.set_only_labeled_data(enable=True)  # the trick to make the pretrain-finetune framework working.

    if two_stage_training:
        trainer.set_train_with_two_stage(enable=True)

    trainer.start_training()


# trainer.inference(checkpoint=checkpoint)
if __name__ == '__main__':
    main()
