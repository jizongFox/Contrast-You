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
                    choices=["Up_conv5", "Up_conv4", "Up_conv3", "Up_conv2", "DeConv_1x1"],
                    help="features from which the contrastive learning are made")
parser.add_argument("--save_dir", required=True, type=str, help="save_dir for the save folder")
parser.add_argument("--time", default=4, type=int, help="demanding time")
parser.add_argument("--lr", default=None, type=str, help="learning rate")
parser.add_argument("--output-size", default=16, help="output size of decoder")
parser.add_argument("--on-local", default=False, action="store_true", help="run on local")
args = parser.parse_args()

num_batches = args.num_batches
random_seed = args.random_seed
max_epoch = args.max_epoch

label_ratio = args.label_ratio

lr: str = args.lr or f"{lr_zooms[args.dataset_name]:.10f}"

save_dir = args.save_dir
features = args.features

importance_weights = [str(1.0) if x == "Conv5" else str(0.5) for x in features]

output_size = int(args.output_size)

SharedParams = f" Data.labeled_data_ratio={label_ratio} " \
               f" Data.unlabeled_data_ratio={1 - label_ratio} " \
               f" Data.name={args.dataset_name}" \
               f" Trainer.max_epoch={max_epoch} " \
               f" Trainer.num_batches={num_batches} " \
               f" Arch.num_classes={dataset_name2class_numbers[args.dataset_name]} " \
               f" RandomSeed={random_seed} "

TrainerParams = SharedParams + f" Optim.lr={lr_zooms[args.dataset_name]:.10f} "

PretrainParams = SharedParams

save_dir += ("/" + "/".join([args.dataset_name, f"label_ratio_{label_ratio}"]))

baselines = [

    # ps using only labeled data
    f"python main_infonce.py {TrainerParams} Trainer.name=finetune Trainer.save_dir={save_dir}/ps ",

    # fs using only labeled data
    f"python main_infonce.py {TrainerParams} Trainer.name=finetune Trainer.save_dir={save_dir}/fs "
    f"                     Data.labeled_data_ratio=1.0 Data.unlabeled_data_ratio=0.0 ",
]

Encoder_jobs = [
    # contrastive learning with pretrain Conv5
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5]"
    f" ProjectorParams.GlobalParams.feature_importance=[1.0]"
    f" Trainer.save_dir={save_dir}/infonce/Conv5_baseline "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Conv5+
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0] "
    f" ProjectorParams.DenseParams.feature_names=[Conv5] "
    f" ProjectorParams.DenseParams.feature_importance=[0.01] "
    f" InfoNCEParameters.DenseParams.include_all=true "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_dense/w_0.01/batchwise "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0] "
    f" ProjectorParams.DenseParams.feature_names=[Conv5] "
    f" ProjectorParams.DenseParams.feature_importance=[0.01] "
    f" InfoNCEParameters.DenseParams.include_all=false "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_dense/w_0.01/imagewise "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Conv5+
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0] "
    f" ProjectorParams.DenseParams.feature_names=[Conv5] "
    f" ProjectorParams.DenseParams.feature_importance=[0.001] "
    f" InfoNCEParameters.DenseParams.include_all=true "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_dense/w_0.001/batchwise "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0] "
    f" ProjectorParams.DenseParams.feature_names=[Conv5] "
    f" ProjectorParams.DenseParams.feature_importance=[0.001] "
    f" InfoNCEParameters.DenseParams.include_all=false "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_dense/w_0.001/imagewise "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Conv5+
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0] "
    f" ProjectorParams.DenseParams.feature_names=[Conv5] "
    f" ProjectorParams.DenseParams.feature_importance=[0.0001] "
    f" InfoNCEParameters.DenseParams.include_all=true "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_dense/w_0.0001/batchwise "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0] "
    f" ProjectorParams.DenseParams.feature_names=[Conv5] "
    f" ProjectorParams.DenseParams.feature_importance=[0.0001] "
    f" InfoNCEParameters.DenseParams.include_all=false "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_dense/w_0.0001/imagewise "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",


    # contrastive learning with pretrain Conv5+
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0] "
    f" ProjectorParams.DenseParams.feature_names=[Conv5] "
    f" ProjectorParams.DenseParams.feature_importance=[0.1] "
    f" InfoNCEParameters.DenseParams.include_all=true "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_dense/w_0.1/batchwise "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0] "
    f" ProjectorParams.DenseParams.feature_names=[Conv5] "
    f" ProjectorParams.DenseParams.feature_importance=[0.1] "
    f" InfoNCEParameters.DenseParams.include_all=false "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_dense/w_0.1/imagewise "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Conv5+
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0] "
    f" ProjectorParams.DenseParams.feature_names=[Conv5] "
    f" ProjectorParams.DenseParams.feature_importance=[1.0] "
    f" InfoNCEParameters.DenseParams.include_all=true "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_dense/w_1.0/batchwise "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0] "
    f" ProjectorParams.DenseParams.feature_names=[Conv5] "
    f" ProjectorParams.DenseParams.feature_importance=[1.0] "
    f" InfoNCEParameters.DenseParams.include_all=false "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_dense/w_1.0/imagewise "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Conv5 and Upconv3 jointly
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Up_conv3] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,0.1] "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_Upconv3/w_1.0_0.1 "
    f" Trainer.grad_from=Conv1 Trainer.grad_to=Up_conv3 "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Conv5 and Upconv3 jointly
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Up_conv3] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,0.01] "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_Upconv3/w_1.0_0.01 "
    f" Trainer.grad_from=Conv1 Trainer.grad_to=Up_conv3 "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Conv5 and Upconv3 jointly
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Up_conv3] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,0.0001] "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_Upconv3/w_1.0_0.0001 "
    f" Trainer.grad_from=Conv1 Trainer.grad_to=Up_conv3 "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

]

Decoder_Jobs = [
    # contrastive learning with pretrain
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  Trainer.save_dir={save_dir}/infonce/{'_'.join(features)}/pretrain "
    f" Arch.checkpoint=runs/{save_dir}/infonce/Conv5/pretrain/last.pth     Trainer.grad_from=Up5        "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce.yaml"
    f"    &&  "
    f"python main_infonce.py {TrainerParams} Trainer.name=finetune          Trainer.save_dir={save_dir}/infonce/{'_'.join(features)}/train "
    f"              Arch.checkpoint=runs/{save_dir}/infonce/{'_'.join(features)}/pretrain/last.pth ",

    # # improved contrastive learning with pretrain
    f"python main_infonce.py {PretrainParams} Trainer.name=experimentpretrain  Trainer.save_dir={save_dir}/new1/{'_'.join(features)}/pretrain "
    f" Arch.checkpoint=runs/{save_dir}/new1/Conv5/pretrain/last.pth        Trainer.grad_from=Up5     "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/new.yaml"
    f"   &&  "
    f"python main_infonce.py {TrainerParams} Trainer.name=finetune  Trainer.save_dir={save_dir}/new1/{'_'.join(features)}/train "
    f"              Arch.checkpoint=runs/{save_dir}/new1/{'_'.join(features)}/pretrain/last.pth "

]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

job_submiter = JobSubmiter(project_path="../../", on_local=args.on_local, time=args.time, )

for j in [*Encoder_jobs, *baselines]:
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
