import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # noqa

parser.add_argument("-n", "--dataset_name", default="acdc", type=str, help="dataset name")
parser.add_argument("-l", "--label_ratio", default=0.05, type=float, help="labeled ratio")
parser.add_argument("-b", "--num_batches", default=500, type=int, help="num batches")
parser.add_argument("-e", "--max_epoch", default=100, type=int, help="max epoch")
parser.add_argument("-s", "--random_seed", default=1, type=int, help="random seed")
parser.add_argument("--save_dir", default=None, type=str, help="save_dir")
parser.add_argument("--time", default=4, type=int, help="demanded time")
parser.add_argument("--distributed", action="store_true", default=False, help="enable distributed training")
parser.add_argument("--num_gpus", default=1, type=int, help="gpu numbers")
parser.add_argument("--two_stage_training", default=False, action="store_true", help="two_stage_training")
parser.add_argument("--disable_bn", default=False, action="store_true", help="disable_bn")
parser.add_argument("--lr", default=None, type=str, help="learning rate")
parser.add_argument("--net", default="unet", choices=["unet", "unetindex"], help="network choice")
args = parser.parse_args()

if args.distributed:
    assert args.num_gpus > 1
else:
    args.num_gpus = 1

if args.disable_bn:
    assert args.two_stage_training

num_batches = args.num_batches
random_seed = args.random_seed

labeled_data_ratio = args.label_ratio
two_stage_training = args.two_stage_training
disable_bn = args.disable_bn
net_name = args.net

dataset_name2class_numbers = {
    "acdc": 4,
    "prostate": 2,
    "spleen": 2,
    "mmwhs": 5,
}
lr_zooms = {"acdc": 0.0000001,
            "prostate": 0.000001,
            "spleen": 0.000001,
            "mmwhs": 0.000001}

save_dir_main = args.save_dir if args.save_dir else "main_result_folder"

lr: str = args.lr or f"{lr_zooms[args.dataset_name]:.10f}"

save_dir = f"{save_dir_main}/{args.dataset_name}/" \
           f"label_data_ration_{labeled_data_ratio}/" \
           f"{'two' if two_stage_training else 'single'}_stage_training/" \
           f"network_{net_name}/" \
           f"bn_track_{str(not disable_bn)}/" \
           f"random_seed_{random_seed}/" \
           f"lr_{lr}"

common_opts = f" Data.labeled_data_ratio={args.label_ratio} " \
              f" Data.unlabeled_data_ratio={1 - args.label_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Trainer.max_epoch={args.max_epoch} " \
              f" Data.name={args.dataset_name} " \
              f" Arch.num_classes={dataset_name2class_numbers[args.dataset_name]} " \
              f" Optim.lr={lr_zooms[args.dataset_name]:.10f} " \
              f" RandomSeed={random_seed} " \
              f" DistributedTrain={args.distributed}" \
              f" Trainer.two_stage_training={two_stage_training} " \
              f" Trainer.disable_bn_track_for_unlabeled_data={disable_bn} " \
              f" Arch.name={net_name} "

jobs = [
    # baseline
    f" python main.py {common_opts} Trainer.name=partial Trainer.save_dir={save_dir}/ps  ",

    f" python main.py {common_opts} Trainer.name=partial Trainer.save_dir={save_dir}/fs "
    f" Data.labeled_data_ratio=1 Data.unlabeled_data_ratio=0",

    # only labeled data
    f" python main.py {common_opts} Trainer.name=partial Trainer.save_dir={save_dir}/ps_only_label "
    f" Trainer.only_labeled_data=true ",

    f" python main.py {common_opts} Trainer.name=partial Trainer.save_dir={save_dir}/fs_only_label "
    f" Data.labeled_data_ratio=1 Data.unlabeled_data_ratio=0 Trainer.only_labeled_data=true",

    # mean teacher
    f" python main.py {common_opts} Trainer.name=meanteacher Trainer.save_dir={save_dir}/mt/1 "
    f" MeanTeacherParameters.weight=1.0 ",

    f" python main.py {common_opts} Trainer.name=meanteacher Trainer.save_dir={save_dir}/mt/5 "
    f" MeanTeacherParameters.weight=5.0 ",

    f" python main.py {common_opts} Trainer.name=meanteacher Trainer.save_dir={save_dir}/mt/10 "
    f" MeanTeacherParameters.weight=10.0 ",

    f" python main.py {common_opts} Trainer.name=meanteacher Trainer.save_dir={save_dir}/mt/15 "
    f" MeanTeacherParameters.weight=15.0 ",
]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

jobsubmiter = JobSubmiter(project_path="../", on_local=False, time=args.time, gres=f"gpu:{args.num_gpus}")
for j in jobs:
    jobsubmiter.prepare_env(
        [
            "source ../venv/bin/activate ",
            "export OMP_NUM_THREADS=1",
            "export PYTHONOPTIMIZE=1"
        ]
    )
    jobsubmiter.account = next(accounts)
    jobsubmiter.run(j)
