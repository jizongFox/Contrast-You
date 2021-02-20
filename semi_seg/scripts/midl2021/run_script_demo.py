import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter
from deepclustering2.utils import gethash
from semi_seg.scripts.helper import dataset_name2class_numbers, lr_zooms

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-n", "--dataset_name", default="acdc", type=str, help="dataset name")
parser.add_argument("-b", "--num_batches", default=500, type=int, help="num batches")
parser.add_argument("-e", "--max_epoch", default=100, type=int, help="max epoch")
parser.add_argument("-s", "--random_seed", default=1, type=int, help="random seed")
parser.add_argument("--group_sample_num", "-g", default=6, type=int, help="group_sample_num for contrastive loader")
parser.add_argument("--save_dir", required=True, type=str, help="save_dir for the save folder")
parser.add_argument("--time", default=4, type=int, help="demanding time")
parser.add_argument("--lr", default=None, type=str, help="learning rate")
parser.add_argument("--on-local", default=False, action="store_true", help="run on local")
args = parser.parse_args()

num_batches = args.num_batches
random_seed = args.random_seed
max_epoch = args.max_epoch
group_sample_num = args.group_sample_num

__githash__ = gethash(__file__)
lr: str = args.lr or f"{lr_zooms[args.dataset_name]:.10f}"

save_dir = args.save_dir

SharedParams = f" Data.name={args.dataset_name}" \
               f" Trainer.max_epoch={max_epoch} " \
               f" Trainer.num_batches={num_batches} " \
               f" Arch.num_classes={dataset_name2class_numbers[args.dataset_name]} " \
               f" RandomSeed={random_seed} "

TrainerParams = SharedParams + f" Optim.lr={lr_zooms[args.dataset_name]:.10f} "

PretrainParams = SharedParams + f" ContrastiveLoaderParams.group_sample_num={group_sample_num}"

save_dir += ("/" + "/".join(
    [f"githash_{__githash__[:7]}", args.dataset_name, f"sample_num_{group_sample_num}", f"random_seed_{random_seed}"]))

baselines = [
    f"python main_finetune.py {TrainerParams} Trainer.name=finetune Trainer.save_dir={save_dir}/baseline ",
]

Encoder_jobs = [
    # contrastive learning with pretrain Conv5
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5]"
    f" ProjectorParams.GlobalParams.feature_importance=[1.0]"
    f" Trainer.save_dir={save_dir}/infonce/Conv5_baseline/encoder "
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

    # contrastive learning with pretrain Conv5+
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0] "
    f" ProjectorParams.DenseParams.feature_names=[Conv5] "
    f" ProjectorParams.DenseParams.feature_importance=[0.001] "
    f" InfoNCEParameters.DenseParams.include_all=true "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_dense/w_0.001/batchwise "
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

    # contrastive learning with pretrain Conv5+
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0] "
    f" ProjectorParams.DenseParams.feature_names=[Conv5] "
    f" ProjectorParams.DenseParams.feature_importance=[0.1] "
    f" InfoNCEParameters.DenseParams.include_all=true "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_dense/w_0.1/batchwise "
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

    # contrastive learning with pretrain Conv5 and Upconv3 jointly
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Up_conv3] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,0.5] "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_Upconv3/w_1.0_0.5 "
    f" Trainer.grad_from=Conv1 Trainer.grad_util=Up_conv3 "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Conv5 and Upconv3 jointly
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Up_conv3] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,0.1] "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_Upconv3/w_1.0_0.1 "
    f" Trainer.grad_from=Conv1 Trainer.grad_util=Up_conv3 "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Conv5 and Upconv3 jointly
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Up_conv3] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,0.01] "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_Upconv3/w_1.0_0.01 "
    f" Trainer.grad_from=Conv1 Trainer.grad_util=Up_conv3 "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Conv5 and Upconv3 jointly
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Up_conv3] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,0.0001] "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_Upconv3/w_1.0_0.0001 "
    f" Trainer.grad_from=Conv1 Trainer.grad_util=Up_conv3 "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Conv5 and Upconv3 jointly
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Up_conv3] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,0.00001] "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_Upconv3/w_1.0_0.00001 "
    f" Trainer.grad_from=Conv1 Trainer.grad_util=Up_conv3 "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Conv5 and Upconv3 jointly
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Up_conv3] "
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,0.0001] "
    f" ProjectorParams.DenseParams.feature_names=[Conv5,] "
    f" ProjectorParams.DenseParams.feature_importance=[0.0001,] "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_dense_Upconv3/w_1.0_0.0001_0.00001 "
    f" Trainer.grad_from=Conv1 Trainer.grad_util=Up_conv3 "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

]

Decoder_Jobs = [
    # contrastive learning with pretrain Decoder that takes encoders from conv5
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[]"
    f" ProjectorParams.GlobalParams.feature_importance=[]"
    f" ProjectorParams.DenseParams.feature_names=[Up_conv3]"
    f" ProjectorParams.DenseParams.feature_importance=[1.0]"
    f" Trainer.grad_from=Up5 Trainer.grad_util=Up_conv3 "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_baseline/decoder_up_conv3 "
    f" Arch.checkpoint=runs/{save_dir}/infonce/Conv5_baseline/encoder/pre/last.pth "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Decoder that takes encoders from conv5
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Up_conv3]"
    f" ProjectorParams.GlobalParams.feature_importance=[1.0]"
    f" ProjectorParams.DenseParams.feature_names=[Up_conv3]"
    f" ProjectorParams.DenseParams.feature_importance=[0.1]"
    f" Trainer.grad_from=Up5 Trainer.grad_util=Up_conv3 "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_baseline/decoder_up_conv3_global "
    f" Arch.checkpoint=runs/{save_dir}/infonce/Conv5_baseline/encoder/pre/last.pth "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",

    # contrastive learning with pretrain Decoder that takes encoders all together
    f"python main_infonce.py {PretrainParams} Trainer.name=infoncepretrain  "
    f" ProjectorParams.GlobalParams.feature_names=[Conv5,Up_conv3]"
    f" ProjectorParams.GlobalParams.feature_importance=[1.0,0.05]"
    f" ProjectorParams.DenseParams.feature_names=[Up_conv3]"
    f" ProjectorParams.DenseParams.feature_importance=[0.05]"
    f" Trainer.grad_from=Conv1 Trainer.grad_util=Up_conv3 "
    f" Trainer.save_dir={save_dir}/infonce/Conv5_baseline/all_decoder_up_conv3_global_dense "
    f" --opt_config_path ../config/specific/pretrain.yaml ../config/specific/infonce2.yaml",


]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

job_submiter = JobSubmiter(project_path="../../", on_local=args.on_local, time=args.time, )

for j in [*Encoder_jobs, *Decoder_Jobs]:
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
