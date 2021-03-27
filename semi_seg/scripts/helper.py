from contextlib import contextmanager
from typing import Dict, Any

import numpy as np
from deepclustering2.utils import gethash, write_yaml

dataset_name2class_numbers = {
    "acdc": 4,
    "prostate": 2,
    "spleen": 2,
    "mmwhs": 5,
}
ft_lr_zooms = {"acdc": 0.0000001,
               "prostate": 0.000001,
               "spleen": 0.000001,
               "mmwhs": 0.000001}

pre_lr_zooms = {"acdc": 0.0000005, }

# CC things
__accounts = ["def-chdesa", "def-mpederso", "rrg-mpederso"]


def account_iterable(name_list):
    while True:
        for i in np.random.permutation(name_list):
            yield i


accounts = account_iterable(__accounts)

__git_hash__ = gethash(__file__)


def _assert_equality(feature_name, importance):
    assert len(feature_name) == len(importance)


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
                               default=["Conv5"], type=str, help="global_features")
        subparser.add_argument("--global_importance", nargs="+", type=float, default=[1.0, ], help="global importance")

        subparser.add_argument("--contrast_on", "-c", nargs="+", type=str, required=True,
                               choices=["partition", "cycle", "patient"])

    def parse(self, args):
        self.add(f"ContrastiveLoaderParams.group_sample_num={args.group_sample_num}")
        _assert_equality(args.global_features, args.global_importance)

        self.add(f"ProjectorParams.GlobalParams.feature_names=[{','.join(args.global_features)}]")
        self.add(
            f"ProjectorParams.GlobalParams.feature_importance=[{','.join([str(x) for x in args.global_importance])}]")
        self.add(f"ProjectorParams.LossParams.contrast_on=[{','.join(args.contrast_on)}]")


class BindSelfPaced(_BindOptions):
    @staticmethod
    def bind(subparser):
        subparser.add_argument("--begin_value", default=[1000], type=float, nargs="+",
                               help="ProjectorParams.LossParams.begin_value")
        subparser.add_argument("--end_value", default=[1000], type=float, nargs="+",
                               help="ProjectorParams.LossParams.end_value")
        subparser.add_argument("--method", default="hard", type=str, nargs="+",
                               help="ProjectorParams.LossParams.weight_update")

    def parse(self, args):
        self.add(f"ProjectorParams.LossParams.begin_value=[{','.join([str(x) for x in args.begin_value])}]")
        self.add(f"ProjectorParams.LossParams.end_value=[{','.join([str(x) for x in args.end_value])}]")
        self.add(f"ProjectorParams.LossParams.weight_update=[{','.join(args.method)}]")


class BindSemiSupervisedLearning(_BindOptions):
    pass


@contextmanager
def dump_config(config: Dict[str, Any]):
    import string
    import random
    import os
    tmp_path = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) + ".yaml"
    write_yaml(config, save_dir="./", save_name=tmp_path, force_overwrite=True)
    tmp_path = os.path.abspath(tmp_path)
    yield tmp_path

    def remove():
        os.remove(tmp_path)

    import atexit
    atexit.register(remove)
