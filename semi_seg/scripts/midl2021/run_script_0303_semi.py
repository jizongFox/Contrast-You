import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter
from deepclustering2.utils import gethash

from semi_seg.scripts.helper import dataset_name2class_numbers, ft_lr_zooms

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-n", "--dataset_name", default="acdc", type=str, help="dataset name")
parser.add_argument("-b", "--num_batches", default=500, type=int, help="num batches")
parser.add_argument("-e", "--max_epoch", default=100, type=int, help="max epoch")
parser.add_argument("-l", "--label_ratio", default=0.01, type=float, help="labeled_ratio")
parser.add_argument("-s", "--random_seed", default=1, type=int, help="random seed")
parser.add_argument("--save_dir", required=True, type=str, help="save_dir for the save folder")
parser.add_argument("--time", default=4, type=int, help="demanding time")
parser.add_argument("--lr", default=None, type=str, help="learning rate")
parser.add_argument("--on-local", default=False, action="store_true", help="run on local")

args = parser.parse_args()

num_batches = args.num_batches
random_seed = args.random_seed
max_epoch = args.max_epoch

__git_hash__ = gethash(__file__)

lr: str = args.lr or f"{ft_lr_zooms[args.dataset_name]:.10f}"

save_dir = args.save_dir

SharedParams = f" Data.name={args.dataset_name}" \
               f" Trainer.max_epoch={max_epoch} " \
               f" Trainer.num_batches={num_batches} " \
               f" Arch.num_classes={dataset_name2class_numbers[args.dataset_name]} " \
               f" RandomSeed={random_seed} " \
               f" Trainer.two_stage_training=true " \
               f" Data.labeled_data_ratio={args.label_ratio} " \
               f" Data.unlabeled_data_ratio={1 - args.label_ratio} "

TrainerParams = SharedParams + f" Optim.lr={ft_lr_zooms[args.dataset_name]:.10f} "

PretrainParams = SharedParams

save_dir += ("/" + "/".join(
    [
        f"githash_{__git_hash__[:7]}",
        args.dataset_name,
        f"random_seed_{random_seed}"
    ]))

baseline = [
    f"python main.py {TrainerParams} Trainer.name=finetune Trainer.save_dir={save_dir}/baseline ",
    f"python main.py {PretrainParams} Trainer.name=infonce  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5]"
    f" ProjectorParams.GlobalParams.feature_importance=[1.0]"
    f" Trainer.save_dir={save_dir}/infonce/conv5/global "
    f" --opt_config_path ../config/specific/infonce2.yaml",
]

# this is the baseline using infonce for global and dense optimization
proposed = [
    # contrastive learning with pretrain Conv5
    f"python main_infonce.py {PretrainParams} Trainer.name=infonce  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Conv5,Conv5]"
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,1.0,1.0]"
    f" InfoNCEParameters.GlobalParams.contrast_on=[partition,patient,cycle]"
    f" InfoNCEParameters.weight=0.1 "
    f" Trainer.save_dir={save_dir}/proposed/conv5/global/weight_0.1 "
    f" --opt_config_path  ../config/specific/infoncemultitask.yaml",

    f"python main_infonce.py {PretrainParams} Trainer.name=infonce  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Conv5,Conv5]"
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,1.0,1.0]"
    f" InfoNCEParameters.GlobalParams.contrast_on=[partition,patient,cycle]"
    f" InfoNCEParameters.weight=0.01 "
    f" Trainer.save_dir={save_dir}/proposed/conv5/global/weight_0.01 "
    f" --opt_config_path  ../config/specific/infoncemultitask.yaml",

    f"python main_infonce.py {PretrainParams} Trainer.name=infonce  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Conv5,Conv5]"
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,1.0,1.0]"
    f" InfoNCEParameters.GlobalParams.contrast_on=[partition,patient,cycle]"
    f" InfoNCEParameters.weight=1.0 "
    f" Trainer.save_dir={save_dir}/proposed/conv5/global/weight_1.0 "
    f" --opt_config_path  ../config/specific/infoncemultitask.yaml",

    f"python main_infonce.py {PretrainParams} Trainer.name=infonce  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Conv5,Conv5]"
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,1.0,1.0]"
    f" InfoNCEParameters.GlobalParams.contrast_on=[partition,patient,cycle]"
    f" InfoNCEParameters.weight=5.0 "
    f" Trainer.save_dir={save_dir}/proposed/conv5/global/weight_5.0 "
    f" --opt_config_path  ../config/specific/infoncemultitask.yaml",

]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

job_submiter = JobSubmiter(project_path="../../", on_local=args.on_local, time=args.time, )

for j in [*baseline, *proposed]:
    job_submiter.prepare_env(
        [
            "source ../venv/bin/activate ",
            "export OMP_NUM_THREADS=1",
            "export PYTHONOPTIMIZE=1",
            # "export LOGURU_LEVEL=INFO"
        ]
    )
    job_submiter.account = next(accounts)
    print(j)
    code = job_submiter.run(j)
    if code != 0:
        raise RuntimeError
