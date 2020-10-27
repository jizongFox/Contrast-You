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
from deepclustering2.utils import gethash
import torch
from deepclustering2.utils import set_benchmark
from semi_seg.trainer import trainer_zoos, InfoNCEPretrainTrainer  # noqa
from semi_seg.dsutils import get_dataloaders
from deepclustering2.ddp import initialize_ddp_environment, convert2syncBN

cur_githash = gethash(__file__)


def main():
    cmanager = ConfigManger(Path(PROJECT_PATH) / "config/semi.yaml")
    config = cmanager.config
    port = random.randint(10000, 60000)
    use_distributed_training = config.get("DistributedTrain")
    if use_distributed_training is True:
        ngpus_per_node = torch.cuda.device_count()
        print(f"Found {ngpus_per_node} per node")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, cmanager, port))
    else:
        main_worker(0, 1, config, cmanager, port)


def main_worker(rank, ngpus_per_node, config, cmanager, port):
    use_distributed_training = config.get("DistributedTrain")

    def set_distributed_training():
        config["Trainer"]["device"] = f"cuda:{rank}"
        config["LabeledData"]["batch_size"] = config["LabeledData"]["batch_size"] // ngpus_per_node
        config["UnlabeledData"]["batch_size"] = config["UnlabeledData"]["batch_size"] // ngpus_per_node
        initialize_ddp_environment(rank, ngpus_per_node, dist_url=f"tcp://localhost:{port}")

    if use_distributed_training:
        set_distributed_training()

    set_benchmark(config.get("RandomSeed", 1))

    labeled_loader, unlabeled_loader, val_loader = get_dataloaders(config)

    model = UNet(**config["Arch"])
    if use_distributed_training:
        model = convert2syncBN(model)
        model = torch.nn.parallel.DistributedDataParallel(model.to(rank), device_ids=[rank])

    pretrain_cmanager = ConfigManger(Path(PROJECT_PATH) / "config/pretrain.yaml", verbose=False)
    pretrain_config = pretrain_cmanager.config["PretrainConfig"]

    is_pretrain = pretrain_config.get("use_pretrain", False)
    checkpoint = config.get("Checkpoint", None)
    if is_pretrain:
        pretrainTrainer = InfoNCEPretrainTrainer(
            model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
            val_loader=val_loader, sup_criterion=KL_div(verbose=False),
            configuration=pretrain_config, save_dir=os.path.join(config["Trainer"]["save_dir"], "pretrain"),
            **{k: v for k, v in pretrain_config["Trainer"].items() if k != "save_dir"}
        )
        pretrainTrainer.init()

        if checkpoint is not None:
            pretrainTrainer.load_state_dict_from_path(
                os.path.join(checkpoint, "pretrain"),
                strict=True
            )
        pretrainTrainer.start_training()

    trainer_name = config["Trainer"].pop("name")
    Trainer = trainer_zoos[trainer_name]

    trainer = Trainer(
        model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
        val_loader=val_loader, sup_criterion=KL_div(verbose=False),
        configuration={**cmanager.config, **{"GITHASH": cur_githash}},
        save_dir=os.path.join(config["Trainer"]["save_dir"], "train"),
        **{k: v for k, v in config["Trainer"].items() if k != "save_dir"}
    )
    trainer.init()

    if checkpoint is not None:
        trainer.load_state_dict_from_path(os.path.join(checkpoint, "train"), strict=True)
    trainer.start_training()


# trainer.inference(checkpoint=checkpoint)
if __name__ == '__main__':
    main()
