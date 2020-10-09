import warnings

warnings.filterwarnings("ignore")
from scipy.sparse import issparse  # noqa

_ = issparse  # noqa

from deepclustering2.loss import KL_div
import random
from pathlib import Path
from contrastyou import PROJECT_PATH
from contrastyou.arch import UNet
import torch.multiprocessing as mp
from contrastyou.dataloader._seg_datset import ContrastBatchSampler  # noqa
from deepclustering2.configparser import ConfigManger
from deepclustering2.utils import gethash
import torch
from deepclustering2.utils import set_benchmark
from semi_seg.trainer import trainer_zoos
from semi_seg.dataloader_helper import get_dataloaders
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

    trainer_name = config["Trainer"].pop("name")
    Trainer = trainer_zoos[trainer_name]

    model = UNet(**config["Arch"])
    if use_distributed_training:
        model = convert2syncBN(model)
        model = torch.nn.parallel.DistributedDataParallel(model.to(rank), device_ids=[rank])

    trainer = Trainer(
        model=model, labeled_loader=iter(labeled_loader), unlabeled_loader=iter(unlabeled_loader),
        val_loader=val_loader, sup_criterion=KL_div(verbose=False),
        configuration={**cmanager.config, **{"GITHASH": cur_githash}},
        **config["Trainer"]
    )
    trainer.init()
    checkpoint = config.get("Checkpoint", None)
    if checkpoint is not None:
        trainer.load_state_dict_from_path(checkpoint, strict=True)
    trainer.start_training()


# trainer.inference(checkpoint=checkpoint)
if __name__ == '__main__':
    main()
