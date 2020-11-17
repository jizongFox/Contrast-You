import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

# this is to kill professor's idea that only the temperature works.

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # noqa

parser.add_argument("-n", "--dataset_name", default="acdc", type=str, help="dataset name")
parser.add_argument("-l", "--label_ratio", default=0.05, type=float, help="labeled data ratio")
parser.add_argument("-b", "--num_batches", default=500, type=int, help="num batches")
parser.add_argument("-e", "--max_epoch", default=100, type=int, help="max epoch")
parser.add_argument("-s", "--random_seed", default=1, type=int, help="random seed")
parser.add_argument("--save_dir", default="tmp", type=str, help="save dir")
parser.add_argument("--time", default=4, type=int, help="request time for CC")
parser.add_argument("--distributed", action="store_true", default=False, help="enable distributed training")
parser.add_argument("--num_gpus", default=1, type=int, help="gpu numbers")
parser.add_argument("--headtypes", default="linear", choices=["linear", "mlp"], help="projector head types")

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
           f"random_seed_{random_seed}/head_type_{args.headtypes}"

common_opts = f" Data.labeled_data_ratio={args.label_ratio} " \
              f" Data.unlabeled_data_ratio={1 - args.label_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Trainer.max_epoch={args.max_epoch} " \
              f" Data.name={args.dataset_name} " \
              f" Arch.num_classes={dataset_name2class_numbers[args.dataset_name]} " \
              f" Optim.lr={lr_zooms[args.dataset_name]:.10f} " \
              f" RandomSeed={random_seed} " \
              f" DistributedTrain={args.distributed} " \
              f" IICRegParameters.EncoderParams.head_types={args.headtypes} " \
              f" IICRegParameters.DecoderParams.head_types={args.headtypes} "

only_encoder_group_opt = " Trainer.feature_names=[Conv5] Trainer.feature_importance=[1.0] "

jobs = [
    # clustering normalizing comparison # encoder+ decoder setting
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/false_flase "
    f" IICRegParameters.EncoderParams.normalize=false "
    f" IICRegParameters.DecoderParams.normalize=false",
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/false_true "
    f" IICRegParameters.EncoderParams.normalize=false "
    f" IICRegParameters.DecoderParams.normalize=true",
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/true_flase "
    f" IICRegParameters.EncoderParams.normalize=true "
    f" IICRegParameters.DecoderParams.normalize=false",
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/true_true "
    f" IICRegParameters.EncoderParams.normalize=true "
    f" IICRegParameters.DecoderParams.normalize=true",

    # clusteirng normalization comparison on only encoder part

    f" python main.py {common_opts + only_encoder_group_opt} "
    f" Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/only_encoder/true "
    f" IICRegParameters.EncoderParams.normalize=true ",

    f" python main.py {common_opts + only_encoder_group_opt} "
    f" Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/only_encoder/false "
    f" IICRegParameters.EncoderParams.normalize=false ",


    # clustering encoder + decoder with temperature
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/temporature/encoder_decoder/1.0 "
    f" IICRegParameters.EncoderParams.temperature=1.0 "
    f" IICRegParameters.DecoderParams.temperature=1.0 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/temporature/encoder_decoder/2.0 "
    f" IICRegParameters.EncoderParams.temperature=2.0 "
    f" IICRegParameters.DecoderParams.temperature=2.0 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/temporature/encoder_decoder/5.0 "
    f" IICRegParameters.EncoderParams.temperature=5.0 "
    f" IICRegParameters.DecoderParams.temperature=5.0 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/temporature/encoder_decoder/10.0 "
    f" IICRegParameters.EncoderParams.temperature=10.0 "
    f" IICRegParameters.DecoderParams.temperature=10.0 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/temporature/encoder_decoder/0.5 "
    f" IICRegParameters.EncoderParams.temperature=0.5 "
    f" IICRegParameters.DecoderParams.temperature=0.5 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/temporature/encoder_decoder/0.2 "
    f" IICRegParameters.EncoderParams.temperature=0.2 "
    f" IICRegParameters.DecoderParams.temperature=0.2 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/temporature/encoder_decoder/0.1 "
    f" IICRegParameters.EncoderParams.temperature=0.1 "
    f" IICRegParameters.DecoderParams.temperature=0.1 ",

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
