import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter
from semi_seg.scripts.helper import dataset_name2class_numbers, lr_zooms

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-n", "--dataset_name", default="acdc", type=str, help="dataset name")
parser.add_argument("-l", "--label_ratio", default=0.05, type=float, help="labeled ratio")
parser.add_argument("-b", "--num_batches", default=500, type=int, help="num batches")
parser.add_argument("-e", "--max_epoch", default=100, type=int, help="max epoch")
parser.add_argument("-s", "--random_seed", default=1, type=int, help="random seed")
parser.add_argument("-f", "--features", required=True, nargs="+",
                    choices=["Conv5", "Up_conv5", "Up_conv4", "Up_conv3", "Up_conv2", "DeConv_1x1"],
                    help="features from which the contrastive learning are made")
parser.add_argument("--save_dir", required=True, type=str, help="save_dir for the save folder")
parser.add_argument("--time", default=4, type=int, help="demanding time")
parser.add_argument("--lr", default=None, type=str, help="learning rate")
parser.add_argument("--output-size", default=20, help="output size of decoder")
args = parser.parse_args()

num_batches = args.num_batches
random_seed = args.random_seed
max_epoch = args.max_epoch

label_ratio = args.label_ratio

save_dir_main = args.save_dir if args.save_dir else "main_result_folder"

lr: str = args.lr or f"{lr_zooms[args.dataset_name]:.10f}"

save_dir = args.save_dir

features = args.features
importance_weights = [str(1.0) if x == "Conv5" else str(0.5) for x in features]
output_size = int(args.output_size)

SharedParams = f" Data.labeled_data_ratio={label_ratio} " \
               f" Data.unlabeled_data_ratio={1 - label_ratio} " \
               f" Trainer.max_epoch={max_epoch} " \
               f" Trainer.num_batches={num_batches} " \
               f" Arch.num_classes={dataset_name2class_numbers[args.dataset_name]} " \
               f" RandomSeed={random_seed} " \
               f" Trainer.feature_names=[{','.join(features)}] " \
               f" Trainer.feature_importance=[{','.join(importance_weights)}] "

SharedTrainerParams = SharedParams + f" Optim.lr={lr_zooms[args.dataset_name]:.10f} "

PretrainParams = SharedParams + f"InfoNCEParameters.DecoderParams.output_size=[{output_size},{output_size}]"

jobs = [
    # ps using one stage trianing
    f"python main.py {SharedTrainerParams} Trainer.name=partial Trainer.save_dir={save_dir}/ps",

    # ps using only labeled data
    f"python main.py {SharedTrainerParams} Trainer.name=partial Trainer.save_dir={save_dir}/ps_only_labeled "
    f"                                     Trainer.only_labeled_data=true",

    # fs using only labeled data
    f"python main.py {SharedTrainerParams} Trainer.name=partial Trainer.save_dir={save_dir}/fs "
    f"                   Trainer.only_labeled_data=true  Data.labeled_data_ratio=1.0 Data.unlabeled_data_ratio=0.0 ",

    # contrastive learning with pretrain
    f"python main.py {PretrainParams} Trainer.name=infoncepretrain  Trainer.save_dir={save_dir}/infonce/pretrain "
    f"               --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce.yaml"
    f"    &&  "
    f"python main.py {SharedTrainerParams} Trainer.name=partial  Trainer.only_labeled_data=true "
    f"              Trainer.save_dir={save_dir}/infonce/train "
    f"              Arch.checkpoint=runs/{save_dir}/infonce/pretrain/last.pth ",

    # # improved contrastive learning with pretrain
    f"python main.py {PretrainParams} Trainer.name=experimentpretrain  Trainer.save_dir={save_dir}/new1/pretrain "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce.yaml"
    f"   &&  "
    f"python main.py {SharedTrainerParams} Trainer.name=partial  Trainer.only_labeled_data=true Trainer.save_dir={save_dir}/new1/train "
    f"              Arch.checkpoint=runs/{save_dir}/new1/pretrain/last.pth "

]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

jobsubmiter = JobSubmiter(project_path="../../", on_local=False, time=args.time, )
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
