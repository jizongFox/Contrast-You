import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter
from deepclustering2.utils import gethash
from semi_seg.scripts.helper import dataset_name2class_numbers, ft_lr_zooms


class _BindOptions:

    def __init__(self) -> None:
        super().__init__()
        self.OptionalScripts = []

    @staticmethod
    def bind(subparser):
        ...

    def parse(self, args):
        ...

    def add(self, string):
        self.OptionalScripts.append(string)

    def get_option_str(self):
        return " ".join(self.OptionalScripts)


class BindPretrainFinetune(_BindOptions):
    @staticmethod
    def bind(subparser):
        subparser.add_argument("--pre_lr", default="null", type=str, help="pretrain learning rate")
        subparser.add_argument("--ft_lr", default="null", type=str, help="finetune learning rate")
        subparser.add_argument("-pe", "--pre_max_epoch", type=str, default="null", help="pretrain max_epoch")
        subparser.add_argument("-fe", "--ft_max_epoch", type=str, default="null", help="finetune max_epoch")

    def parse(self, args):
        pre_lr = args.pre_lr
        self.add(f"Optim.pre_lr={pre_lr}")
        ft_lr = args.ft_lr
        self.add(f"Optim.ft_lr={ft_lr}")
        pre_max_epoch = args.pre_max_epoch
        ft_max_epoch = args.ft_max_epoch
        self.add(f"Trainer.pre_max_epoch={pre_max_epoch}")
        self.add(f"Trainer.ft_max_epoch={ft_max_epoch}")


class BindContrastive(_BindOptions):
    @staticmethod
    def bind(subparser):
        subparser.add_argument("-g", "--group_sample_num", default=6, type=int)
        subparser.add_argument("--global_features", nargs="+", choices=["Conv5", "Conv4", "Conv3", "Conv2"],
                               default=["Conv5"],
                               type=str, help="global_features")
        subparser.add_argument("--global_importance", nargs="+", type=float, default=[1.0, ], help="global importance")

        subparser.add_argument("--dense_features", nargs="+", choices=["Conv5", "Conv4", "Conv3", "Conv2"],
                               default=[],
                               type=str, help="dense_features")
        subparser.add_argument("--dense_importance", nargs="+", type=float, default=[], help="dense importance")

    def parse(self, args):
        group_sample_num = args.group_sample_num
        self.add(f"ContrastiveLoaderParams.group_sample_num={group_sample_num}")
        gfeature_name = args.global_features
        gimportance = args.global_importance
        dfeature_name = args.dense_features
        dimportance = args.dense_importance
        _assert_equality(gfeature_name, gimportance)
        _assert_equality(dfeature_name, dimportance)

        _gfeature_name = ",".join(gfeature_name)
        _gimportance = ",".join([str(x) for x in gimportance])
        _dfeature_name = ",".join(dfeature_name)
        _dimportance = ",".join([str(x) for x in dimportance])

        self.add(f"ProjectorParams.GlobalParams.feature_names=[{_gfeature_name}]")
        self.add(f"ProjectorParams.GlobalParams.feature_importance=[{_gimportance}]")
        self.add(f"ProjectorParams.DenseParams.feature_names=[{_dfeature_name}]")
        self.add(f"ProjectorParams.DenseParams.feature_importance=[{_dimportance}]")


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

comm_parser = parser.add_argument_group("common options")

comm_parser.add_argument("-n", "--dataset_name", default="acdc", type=str, help="dataset name")
comm_parser.add_argument("-b", "--num_batches", default=200, type=int, help="num batches")
comm_parser.add_argument("-s", "--random_seed", default=1, type=int, help="random seed")
comm_parser.add_argument("--save_dir", required=True, type=str, help="save_dir for the save folder")
comm_parser.add_argument("--on-local", default=False, action="store_true", help="run on local")
comm_parser.add_argument("--time", type=int, default=4, help="submitted time to CC")
comm_parser.add_argument("--show_cmd", default=False, action="store_true", help="only show generated cmd.")

subparser = parser.add_subparsers(dest='stage')
baseline = subparser.add_parser("baseline")
infonce = subparser.add_parser("infonce")
softeninfonce = subparser.add_parser("softeninfonce")
mixup = subparser.add_parser("mixup")
multitask = subparser.add_parser("multitask")

# baseline
baseline.add_argument("-e", "--max_epoch", type=str, default=75, help="max_epoch")
baseline.add_argument("--lr", type=str, default=None, help="learning rate")

# infonce
BindPretrainFinetune.bind(infonce)
BindContrastive.bind(infonce)

# soften infonce
BindPretrainFinetune.bind(softeninfonce)
BindContrastive.bind(softeninfonce)
softeninfonce.add_argument("--softenweight", nargs="+", type=float, default=[0.005, ],
                           help="weight between hard and soften")

# mixup
BindPretrainFinetune.bind(mixup)
BindContrastive.bind(mixup)
mixup.add_argument("--softenweight", nargs="+", type=float, default=[0.005, ],
                   help="weight between hard and soften")

# multitask
BindPretrainFinetune.bind(multitask)
BindContrastive.bind(multitask)
multitask.add_argument("--contrast_on", type=str, nargs="+", required=True, choices=["partition", "patient", "cycle"])
args = parser.parse_args()

# setting common params
__git_hash__ = gethash(__file__)
dataset_name = args.dataset_name
num_batches = args.num_batches
random_seed = args.random_seed
num_classes = dataset_name2class_numbers[args.dataset_name]
save_dir = args.save_dir

save_dir += ("/" + "/".join(
    [
        f"githash_{__git_hash__[:7]}",
        args.dataset_name,
        f"random_seed_{random_seed}"
    ]))

SharedParams = f" Data.name={dataset_name}" \
               f" Trainer.num_batches={num_batches} " \
               f" Arch.num_classes={num_classes} " \
               f" RandomSeed={random_seed} "


def _assert_equality(feature_name, importance):
    assert len(feature_name) == len(importance)


if args.stage == "baseline":
    max_epoch = args.max_epoch
    lr = args.lr or f"{ft_lr_zooms[args.dataset_name]:.10f}"
    job_array = [
        f"python main_finetune.py {SharedParams} Optim.lr={lr} Trainer.max_epoch={max_epoch} "
        f"Trainer.name=finetune Trainer.save_dir={save_dir}/baseline "
    ]

elif args.stage == "infonce":
    parser1 = BindPretrainFinetune()
    parser1.parse(args)
    parser2 = BindContrastive()
    parser2.parse(args)

    group_sample_num = args.group_sample_num
    gfeature_names = args.global_features
    gimportance = args.global_importance
    dfeature_names = args.dense_features
    dimportance = args.dense_importance

    save_dir += f"/sample_num_{group_sample_num}"

    subpath = f"global_{'_'.join([*gfeature_names, *[str(x) for x in gimportance]])}/" \
              f"dense_{'_'.join([*dfeature_names, *[str(x) for x in dimportance]])}"
    string = f"python main_infonce.py Trainer.name=infoncepretrain {SharedParams} {parser1.get_option_str()} " \
             f" {parser2.get_option_str()} " \
             f" Trainer.save_dir={save_dir}/infonce/{subpath}/ " \
             f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml"

    job_array = [string]

elif args.stage == "softeninfonce":
    parser1 = BindPretrainFinetune()
    parser1.parse(args)
    parser2 = BindContrastive()
    parser2.parse(args)

    group_sample_num = args.group_sample_num
    gfeature_names = args.global_features
    gimportance = args.global_importance
    dfeature_names = args.dense_features
    dimportance = args.dense_importance
    weights = args.softenweight

    save_dir += f"/sample_num_{group_sample_num}"


    def _infonce_script(weight):
        subpath = f"global_{'_'.join([*gfeature_names, *[str(x) for x in gimportance]])}/" \
                  f"dense_{'_'.join([*dfeature_names, *[str(x) for x in dimportance]])}/" \
                  f"weight_{str(weight)}"

        string = f"python main_infonce.py {SharedParams} Trainer.name=experimentpretrain " \
                 f"{parser1.get_option_str()} {parser2.get_option_str()}" \
                 f" ProjectorParams.GlobalParams.softweight={str(weight)} " \
                 f" ProjectorParams.DenseParams.softweight={str(weight)} " \
                 f" Trainer.save_dir={save_dir}/softeninfonce/{subpath}/ " \
                 f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/new.yaml"
        return string


    job_array = [_infonce_script(w) for w in weights]

elif args.stage == "mixup":
    raise NotImplementedError(args.stage)

elif args.stage == "multitask":
    parser1 = BindPretrainFinetune()
    parser1.parse(args)
    parser2 = BindContrastive()
    parser2.parse(args)

    group_sample_num = args.group_sample_num
    gfeature_names = args.global_features
    gimportance = args.global_importance
    dfeature_names = args.dense_features
    dimportance = args.dense_importance
    contrast_on = args.contrast_on
    assert len(gfeature_names) == len(contrast_on)

    save_dir += f"/sample_num_{group_sample_num}/" \
                f"contrast_on_{'_'.join(contrast_on)}"

    subpath = f"global_{'_'.join([*gfeature_names, *[str(x) for x in gimportance]])}/" \
              f"dense_{'_'.join([*dfeature_names, *[str(x) for x in dimportance]])}"
    string = f"python main_infonce.py Trainer.name=experimentmultitaskpretrain" \
             f" InfoNCEParameters.GlobalParams.contrast_on=[{','.join(contrast_on)}] " \
             f" {SharedParams} {parser1.get_option_str()} " \
             f" {parser2.get_option_str()} " \
             f" Trainer.save_dir={save_dir}/infonce/{subpath}/ " \
             f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infoncemultitask.yaml"

    job_array = [string]

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
            "export CUBLAS_WORKSPACE_CONFIG=:16:8 ",
            # "export LOGURU_LEVEL=INFO"
        ]
    )
    job_submiter.account = next(accounts)
    print(j)
    if not args.show_cmd:
        code = job_submiter.run(j)
        if code != 0:
            raise RuntimeError
