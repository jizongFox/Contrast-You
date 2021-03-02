import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter
from deepclustering2.utils import gethash

from semi_seg.scripts.helper import dataset_name2class_numbers, lr_zooms

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

comm_parser = parser.add_argument_group("common options")

comm_parser.add_argument("-n", "--dataset_name", default="acdc", type=str, help="dataset name")
comm_parser.add_argument("-b", "--num_batches", default=500, type=int, help="num batches")
comm_parser.add_argument("-e", "--max_epoch", default=100, type=int, help="max epoch")
comm_parser.add_argument("-s", "--random_seed", default=1, type=int, help="random seed")
comm_parser.add_argument("--save_dir", required=True, type=str, help="save_dir for the save folder")
comm_parser.add_argument("--lr", default=None, type=str, help="learning rate")
comm_parser.add_argument("--on-local", default=False, action="store_true", help="run on local")
comm_parser.add_argument("--time", type=int, default=4, help="submitted time to CC")
comm_parser.add_argument("--show_cmd",  default=False, action="store_true", help="only show generated cmd.")

subparser = parser.add_subparsers(dest='stage')
baseline = subparser.add_parser("baseline")
infonce = subparser.add_parser("infonce")
softeninfonce = subparser.add_parser("softeninfonce")
mixup = subparser.add_parser("mixup")
multitask = subparser.add_parser("multitask")

# baseline


# infonce
infonce.add_argument("-g", "--group_sample_num", default=6, type=int)
infonce.add_argument("--global_features", nargs="+", choices=["Conv5", "Conv4", "Conv3", "Conv2"], default=["Conv5"],
                     type=str, help="global_features")
infonce.add_argument("--global_importance", nargs="+", type=float, default=[1.0, ], help="global importance")

infonce.add_argument("--dense_features", nargs="+", choices=["Conv5", "Conv4", "Conv3", "Conv2"], default=[],
                     type=str, help="dense_features")
infonce.add_argument("--dense_importance", nargs="+", type=float, default=[], help="dense importance")

# soften infonce
softeninfonce.add_argument("-g", "--group_sample_num", default=6, type=int)
softeninfonce.add_argument("--global_features", nargs="+", choices=["Conv5", "Conv4", "Conv3", "Conv2"],
                           default=["Conv5"],
                           type=str, help="global_features")
softeninfonce.add_argument("--global_importance", nargs="+", type=float, default=[1.0, ], help="global importance")

softeninfonce.add_argument("--dense_features", nargs="+", choices=["Conv5", "Conv4", "Conv3", "Conv2"],
                           default=[],
                           type=str, help="dense_features")
softeninfonce.add_argument("--dense_importance", nargs="+", type=float, default=[], help="dense importance")
softeninfonce.add_argument("--softenweight", nargs="+", type=float, default=[0.005, ],
                           help="weight between hard and soften")

# mixup
mixup.add_argument("-g", "--group_sample_num", default=6, type=int)
mixup.add_argument("--global_features", nargs="+", choices=["Conv5", "Conv4", "Conv3", "Conv2"],
                   default=["Conv5"],
                   type=str, help="global_features")
mixup.add_argument("--global_importance", nargs="+", type=float, default=[1.0, ], help="global importance")

mixup.add_argument("--dense_features", nargs="+", choices=["Conv5", "Conv4", "Conv3", "Conv2"],
                   default=[],
                   type=str, help="dense_features")
mixup.add_argument("--dense_importance", nargs="+", type=float, default=[], help="dense importance")
mixup.add_argument("--softenweight", nargs="+", type=float, default=[0.005, ],
                   help="weight between hard and soften")

# multitask
multitask.add_argument("-g", "--group_sample_num", default=6, type=int)
multitask.add_argument("--global_features", nargs="+", choices=["Conv5", "Conv4", "Conv3", "Conv2"],
                       default=["Conv5"], type=str, help="global_features")
multitask.add_argument("--global_importance", nargs="+", type=float, default=[1.0, ], help="global importance")
multitask.add_argument("--contrast_on", type=str, nargs="+", required=True, choices=["partition", "patient", "both"])

args = parser.parse_args()

# setting common params
num_batches = args.num_batches
random_seed = args.random_seed
max_epoch = args.max_epoch
dataset_name = args.dataset_name
lr: str = args.lr or f"{lr_zooms[args.dataset_name]:.10f}"
num_classes = dataset_name2class_numbers[args.dataset_name]
__git_hash__ = gethash(__file__)

save_dir = args.save_dir

save_dir += ("/" + "/".join(
    [
        f"githash_{__git_hash__[:7]}",
        args.dataset_name,
        f"random_seed_{random_seed}"
    ]))

SharedParams = f" Data.name={dataset_name}" \
               f" Trainer.max_epoch={max_epoch} " \
               f" Trainer.num_batches={num_batches} " \
               f" Arch.num_classes={num_classes} " \
               f" RandomSeed={random_seed} "

TrainerParams = SharedParams + f" Optim.lr={lr_zooms[args.dataset_name]:.10f} "


def _assert_equality(feature_name, importance):
    assert len(feature_name) == len(importance)


if args.stage == "baseline":
    job_array = [
        f"python main_finetune.py {TrainerParams} Trainer.name=finetune Trainer.save_dir={save_dir}/baseline "
    ]

elif args.stage == "infonce":

    group_sample_num = args.group_sample_num
    gfeature = args.global_features
    gimportance = args.global_importance
    dfeature = args.dense_features
    dimportance = args.dense_importance

    InfoNCEParams = SharedParams + f" ContrastiveLoaderParams.group_sample_num={group_sample_num} " \
                                   f" Trainer.name=infoncepretrain "
    save_dir += f"/sample_num_{group_sample_num}"


    def _infonce_script(InfoNCEParams, gfeature_name, gimportance, dfeature_name, dimportance):
        _assert_equality(gfeature_name, gimportance)
        _assert_equality(dfeature_name, dimportance)

        gfeature_name_ = ",".join(gfeature_name)
        gimportance_ = ",".join([str(x) for x in gimportance])
        dfeature_name_ = ",".join(dfeature_name)
        dimportance_ = ",".join([str(x) for x in dimportance])
        subpath = f"global_{'_'.join([*gfeature_name, *[str(x) for x in gimportance]])}/" \
                  f"dense_{'_'.join([*dfeature_name, *[str(x) for x in dimportance]])}"

        string = f"python main_infonce.py {InfoNCEParams} " \
                 f" ProjectorParams.GlobalParams.feature_names=[{gfeature_name_}]" \
                 f" ProjectorParams.GlobalParams.feature_importance=[{gimportance_}]" \
                 f" ProjectorParams.DenseParams.feature_names=[{dfeature_name_}] " \
                 f" ProjectorParams.DenseParams.feature_importance=[{dimportance_}] " \
                 f" Trainer.save_dir={save_dir}/infonce/{subpath}/ " \
                 f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml"
        return string


    job_array = [_infonce_script(InfoNCEParams, gfeature, gimportance, dfeature, dimportance)]

elif args.stage == "softeninfonce":
    group_sample_num = args.group_sample_num
    gfeature = args.global_features
    gimportance = args.global_importance
    dfeature = args.dense_features
    dimportance = args.dense_importance
    weights = args.softenweight

    InfoNCEParams = SharedParams + f" ContrastiveLoaderParams.group_sample_num={group_sample_num} " \
                                   f" Trainer.name=experimentpretrain "
    save_dir += f"/sample_num_{group_sample_num}"


    def _infonce_script(InfoNCEParams, gfeature_name, gimportance, dfeature_name, dimportance, weight):
        _assert_equality(gfeature_name, gimportance)
        _assert_equality(dfeature_name, dimportance)
        gfeature_name_ = ",".join(gfeature_name)
        gimportance_ = ",".join([str(x) for x in gimportance])
        dfeature_name_ = ",".join(dfeature_name)
        dimportance_ = ",".join([str(x) for x in dimportance])
        subpath = f"global_{'_'.join([*gfeature_name, *[str(x) for x in gimportance]])}/" \
                  f"dense_{'_'.join([*dfeature_name, *[str(x) for x in dimportance]])}/" \
                  f"weight_{str(weight)}"

        string = f"python main_infonce.py {InfoNCEParams} " \
                 f" ProjectorParams.GlobalParams.feature_names=[{gfeature_name_}]" \
                 f" ProjectorParams.GlobalParams.feature_importance=[{gimportance_}]" \
                 f" ProjectorParams.DenseParams.feature_names=[{dfeature_name_}] " \
                 f" ProjectorParams.DenseParams.feature_importance=[{dimportance_}] " \
                 f" ProjectorParams.GlobalParams.softweight={str(weight)} " \
                 f" ProjectorParams.DenseParams.softweight={str(weight)} " \
                 f" Trainer.save_dir={save_dir}/softeninfonce/{subpath}/ " \
                 f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/new.yaml"
        return string


    job_array = [_infonce_script(InfoNCEParams, gfeature, gimportance, dfeature, dimportance, w) for w in weights]

elif args.stage == "mixup":
    group_sample_num = args.group_sample_num
    gfeature = args.global_features
    gimportance = args.global_importance
    dfeature = args.dense_features
    dimportance = args.dense_importance
    weights = args.softenweight

    InfoNCEParams = SharedParams + f" ContrastiveLoaderParams.group_sample_num={group_sample_num} " \
                                   f" Trainer.name=experimentmixuppretrain "
    save_dir += f"/sample_num_{group_sample_num}"


    def _infonce_script(InfoNCEParams, gfeature_name, gimportance, dfeature_name, dimportance, weight):
        _assert_equality(gfeature_name, gimportance)
        _assert_equality(dfeature_name, dimportance)
        gfeature_name_ = ",".join(gfeature_name)
        gimportance_ = ",".join([str(x) for x in gimportance])
        dfeature_name_ = ",".join(dfeature_name)
        dimportance_ = ",".join([str(x) for x in dimportance])
        subpath = f"global_{'_'.join([*gfeature_name, *[str(x) for x in gimportance]])}/" \
                  f"dense_{'_'.join([*dfeature_name, *[str(x) for x in dimportance]])}/" \
                  f"weight_{str(weight)}"

        string = f"python main_infonce.py {InfoNCEParams} " \
                 f" ProjectorParams.GlobalParams.feature_names=[{gfeature_name_}]" \
                 f" ProjectorParams.GlobalParams.feature_importance=[{gimportance_}]" \
                 f" ProjectorParams.DenseParams.feature_names=[{dfeature_name_}] " \
                 f" ProjectorParams.DenseParams.feature_importance=[{dimportance_}] " \
                 f" ProjectorParams.GlobalParams.softweight={str(weight)} " \
                 f" ProjectorParams.DenseParams.softweight={str(weight)} " \
                 f" Trainer.save_dir={save_dir}/mixup/{subpath}/ " \
                 f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/new.yaml"
        return string


    job_array = [_infonce_script(InfoNCEParams, gfeature, gimportance, dfeature, dimportance, w) for w in weights]

elif args.stage == "multitask":
    group_sample_num = args.group_sample_num
    gfeature = args.global_features
    gimportance = args.global_importance
    dfeature = []
    dimportance = []
    contrast_on = args.contrast_on
    assert len(gfeature) == len(contrast_on)

    InfoNCEParams = SharedParams + f" ContrastiveLoaderParams.group_sample_num={group_sample_num} " \
                                   f" Trainer.name=experimentmultitaskpretrain " \
                                   f" InfoNCEParameters.GlobalParams.contrast_on=[{','.join(contrast_on)}] "
    save_dir += f"/sample_num_{group_sample_num}"


    def _infonce_script(InfoNCEParams, gfeature_name, gimportance, dfeature_name, dimportance, ):
        _assert_equality(gfeature_name, gimportance)
        _assert_equality(dfeature_name, dimportance)
        gfeature_name_ = ",".join(gfeature_name)
        gimportance_ = ",".join([str(x) for x in gimportance])
        dfeature_name_ = ",".join(dfeature_name)
        dimportance_ = ",".join([str(x) for x in dimportance])
        subpath = f"global_{'_'.join([*gfeature_name, *[str(x) for x in gimportance]])}/" \
                  f"dense_{'_'.join([*dfeature_name, *[str(x) for x in dimportance]])}"

        string = f"python main_infonce.py {InfoNCEParams} " \
                 f" ProjectorParams.GlobalParams.feature_names=[{gfeature_name_}]" \
                 f" ProjectorParams.GlobalParams.feature_importance=[{gimportance_}]" \
                 f" ProjectorParams.DenseParams.feature_names=[{dfeature_name_}] " \
                 f" ProjectorParams.DenseParams.feature_importance=[{dimportance_}] " \
                 f" Trainer.save_dir={save_dir}/multitask/{subpath}/ " \
                 f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infoncemultitask.yaml"
        return string


    job_array = [_infonce_script(InfoNCEParams, gfeature, gimportance, dfeature, dimportance)]

else:
    raise NotImplementedError(args.stage)

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

job_submiter = JobSubmiter(project_path="../../", on_local=args.on_local, time=args.time, )

for j in job_array:
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
    if not args.show_cmd:
        code = job_submiter.run(j)
        if code != 0:
            raise RuntimeError
