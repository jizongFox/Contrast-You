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
                               default=["Conv5"],
                               type=str, help="global_features")
        subparser.add_argument("--global_importance", nargs="+", type=float, default=[1.0, ], help="global importance")

        subparser.add_argument("--dense_features", nargs="+", choices=["Conv5", "Conv4", "Conv3", "Conv2"],
                               default=[],
                               type=str, help="dense_features")
        subparser.add_argument("--dense_importance", nargs="+", type=float, default=[], help="dense importance")
        subparser.add_argument("--exclude_pos", action="store_true", default=False,
                               help="exclude other pos examples to debias contrastive learning")

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

        exclude_pos = "true" if args.exclude_pos else "false"
        self.add(f"InfoNCEParameters.LossParams.exclude_other_pos={exclude_pos}")


class BindSelfPaced(_BindOptions):
    @staticmethod
    def bind(subparser):
        subparser.add_argument("--begin_value", default=4.0, type=float, help="SelfPacedParams.begin_value")
        subparser.add_argument("--end_value", default=16, type=float, help="SelfPacedParams.end_value")
        subparser.add_argument("--method", default="hard", type=str, help="SelfPacedParams.method")

    def parse(self, args):
        begin_value = args.begin_value
        end_value = args.end_value
        method = args.method
        self.add(f"SelfPacedParams.begin_value={begin_value}")
        self.add(f"SelfPacedParams.end_value={end_value}")
        self.add(f"SelfPacedParams.method={method}")
