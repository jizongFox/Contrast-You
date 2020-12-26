import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

# this script is to tell the story that the infonce is not going to work in a supervised-regularization framework.

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # noqa

parser.add_argument("-n", "--dataset_name", default="acdc", type=str)
parser.add_argument("-l", "--label_ratio", default=0.05, type=float)
parser.add_argument("-b", "--num_batches", default=500, type=int)
parser.add_argument("-e", "--max_epoch", default=100, type=int)
parser.add_argument("-s", "--random_seed", default=1, type=int)
parser.add_argument("--save_dir", default=None, type=str)
parser.add_argument("--time", default=4, type=int)
parser.add_argument("--distributed", action="store_true", default=False, help="enable distributed training")
parser.add_argument("--num_gpus", default=1, type=int, help="gpu numbers")

args = parser.parse_args()

if args.distributed:
    assert args.num_gpus > 1, (args.distributed, args.num_gpus)
else:
    args.num_gpus = 1

num_batches = args.num_batches
random_seed = args.random_seed

labeled_data_ratio = args.label_ratio

dataset_name2class_numbers = {
    "acdc": 4,
    "prostate": 2,
    "spleen": 2,
}
lr_zooms = {"acdc": 0.0000001,
            "prostate": 0.000001,
            "spleen": 0.000001}

save_dir_main = args.save_dir
save_dir = f"{save_dir_main}/{args.dataset_name}/" \
           f"label_data_ration_{labeled_data_ratio}/" \
           f"random_seed_{random_seed}"

common_opts = f" Data.labeled_data_ratio={args.label_ratio} " \
              f" Data.unlabeled_data_ratio={1 - args.label_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Trainer.max_epoch={args.max_epoch} " \
              f" Data.name={args.dataset_name} " \
              f" Arch.num_classes={dataset_name2class_numbers[args.dataset_name]} " \
              f" Optim.lr={lr_zooms[args.dataset_name]:.10f} " \
              f" RandomSeed={random_seed} " \
              f" DistributedTrain={args.distributed}"

semi_group_opt = " Trainer.feature_names=[Conv5] Trainer.feature_importance=[1.0] "
no_normalize_opt = " InfoNCEParameters.EncoderParams.normalize=false InfoNCEParameters.DecoderParams.normalize=false  "

jobs = [
    # baseline
    f" python main.py {common_opts} Trainer.name=understandps Trainer.save_dir={save_dir}/infoNCE/ps_only_labeled_data  "
    f" Trainer.only_labeled_data=True",

    f" python main.py {common_opts} Trainer.name=understandps Trainer.save_dir={save_dir}/infoNCE/ps_one_stage_train  "
    f" Trainer.only_labeled_data=False Trainer.two_stage_training=False ",

    f" python main.py {common_opts} Trainer.name=understandps Trainer.save_dir={save_dir}/infoNCE/ps_two_stage_train  "
    f" Trainer.only_labeled_data=False Trainer.two_stage_training=True ",

    f" python main.py {common_opts} Trainer.name=understandps Trainer.save_dir={save_dir}/infoNCE/fs "
    f" Data.labeled_data_ratio=1 Data.unlabeled_data_ratio=0",

]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

jobsubmiter = JobSubmiter(project_path="../../", on_local=False, time=args.time, gres=f"gpu:{args.num_gpus}")
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
