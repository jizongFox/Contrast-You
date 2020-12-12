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
parser.add_argument("--features", required=True, choices=["1", "3"])

args = parser.parse_args()

if args.distributed:
    assert args.num_gpus > 1
else:
    args.num_gpus = 1

num_batches = args.num_batches
random_seed = args.random_seed

labeled_data_ratio = args.label_ratio
two_stage_training = args.two_stage_training

dataset_name2class_numbers = {
    "acdc": 4,
    "prostate": 2,
    "spleen": 2,
}
lr_zooms = {"acdc": 0.0000001,
            "prostate": 0.000001,
            "spleen": 0.000001}

save_dir_main = args.save_dir if args.save_dir else "main_result_folder"

feature_position = "[Conv5,Up_conv3,Up_conv2]" if args.features == "3" else "[Conv5]"
feature_importance = f"[{','.join(['1' for _ in range(len(feature_position.split(',')))])}]"

save_dir = f"{save_dir_main}/{args.dataset_name}/" \
           f"label_data_ration_{labeled_data_ratio}/" \
           f"{'two' if two_stage_training else 'single'}_stage_training/" \
           f"random_seed_{random_seed}/" \
           f"{feature_position.replace('[', '').replace(']', '').replace(',', '_')}"

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
              f" Trainer.feature_names={feature_position} " \
              f" Trainer.feature_importance={feature_importance}"

pretrain_opts = f" PretrainConfig.Trainer.feature_names={feature_position} " \
                f" PretrainConfig.Trainer.feature_importance={feature_importance} "

jobs = [
    # without pretrain
    f" python main.py {common_opts} Trainer.name=partial Trainer.save_dir={save_dir}/ps "
    f" PretrainConfig.Trainer.name=null",

    # pretrain using infonce and train with partial
    f" python main.py {common_opts + pretrain_opts} Trainer.name=partial Trainer.save_dir={save_dir}/pretrain/infonce "
    f" PretrainConfig.Trainer.name=infoncepretrain",

    # pretrain using iic and train with partial
    f" python main.py {common_opts + pretrain_opts} Trainer.name=partial Trainer.save_dir={save_dir}/pretrain/iic "
    f" PretrainConfig.Trainer.name=iicpretrain",

    # pretrain using iicuda and train with partial
    f" python main.py {common_opts + pretrain_opts} Trainer.name=partial Trainer.save_dir={save_dir}/pretrain/udaiic/10_0.1 "
    f" PretrainConfig.Trainer.name=udaiicpretrain PretrainConfig.IICRegParameters.weight=0.1 PretrainConfig.UDARegCriterion.weight=10",

    f" python main.py {common_opts + pretrain_opts} Trainer.name=partial Trainer.save_dir={save_dir}/pretrain/udaiic/5_0.1 "
    f" PretrainConfig.Trainer.name=udaiicpretrain PretrainConfig.IICRegParameters.weight=0.1 PretrainConfig.UDARegCriterion.weight=5",

    f" python main.py {common_opts + pretrain_opts} Trainer.name=partial Trainer.save_dir={save_dir}/pretrain/udaiic/1_0.1 "
    f" PretrainConfig.Trainer.name=udaiicpretrain PretrainConfig.IICRegParameters.weight=0.1 PretrainConfig.UDARegCriterion.weight=1",

    f" python main.py {common_opts + pretrain_opts} Trainer.name=partial Trainer.save_dir={save_dir}/pretrain/udaiic/0.1_0.1 "
    f" PretrainConfig.Trainer.name=udaiicpretrain PretrainConfig.IICRegParameters.weight=0.1 PretrainConfig.UDARegCriterion.weight=0.1",

    f" python main.py {common_opts + pretrain_opts} Trainer.name=partial Trainer.save_dir={save_dir}/pretrain/udaiic/10_0.1 "
    f" PretrainConfig.Trainer.name=udaiicpretrain PretrainConfig.IICRegParameters.weight=0.1 PretrainConfig.UDARegCriterion.weight=10",
    
    # repeat udaiic
    f" python main.py {common_opts + pretrain_opts} Trainer.name=partial Trainer.save_dir={save_dir}/pretrain/udaiic/5_1 "
    f" PretrainConfig.Trainer.name=udaiicpretrain PretrainConfig.IICRegParameters.weight=1 PretrainConfig.UDARegCriterion.weight=5",

    f" python main.py {common_opts + pretrain_opts} Trainer.name=partial Trainer.save_dir={save_dir}/pretrain/udaiic/1_1 "
    f" PretrainConfig.Trainer.name=udaiicpretrain PretrainConfig.IICRegParameters.weight=1 PretrainConfig.UDARegCriterion.weight=1",

    f" python main.py {common_opts + pretrain_opts} Trainer.name=partial Trainer.save_dir={save_dir}/pretrain/udaiic/0.1_1 "
    f" PretrainConfig.Trainer.name=udaiicpretrain PretrainConfig.IICRegParameters.weight=1 PretrainConfig.UDARegCriterion.weight=0.1",
    
    f" python main.py {common_opts + pretrain_opts} Trainer.name=partial Trainer.save_dir={save_dir}/pretrain/udaiic/10_1 "
    f" PretrainConfig.Trainer.name=udaiicpretrain PretrainConfig.IICRegParameters.weight=1 PretrainConfig.UDARegCriterion.weight=10",
    
    

    # train normally
    f"python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/normal/udaiic/5_0.1 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 ",

    f"python main.py {common_opts} Trainer.name=iic Trainer.save_dir={save_dir}/normal/iic/0.1 "
    f" IICRegParameters.weight=0.1  ",

    f"python main.py {common_opts} Trainer.name=iic Trainer.save_dir={save_dir}/normal/iic/0.5 "
    f" IICRegParameters.weight=0.5  "

]

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
    # print(j)