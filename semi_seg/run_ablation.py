import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-l", "--label_ratio", default=0.05, type=float)
parser.add_argument("-b", "--num_batches", default=300, type=int)
parser.add_argument("-e", "--max_epoch", default=100, type=int)
parser.add_argument("-s", "--random_seed", default=1, type=int)
parser.add_argument("-f", "--feature_position", nargs="+", type=str, default=["Conv5", "Up_conv3", "Up_conv2"])
parser.add_argument("--encoder_cluster_subhead", nargs=2, type=int, default=[10, 5])
parser.add_argument("--decoder_cluster_subhead", nargs=2, type=int, default=[10, 5])
parser.add_argument("--decoder_loss_padding_patchsize", nargs=2, type=int, default=[1, 512])
parser.add_argument("--save_dir", default=None, type=str)
parser.add_argument("--time", default=4, type=int)

args = parser.parse_args()

num_batches = args.num_batches
random_seed = args.random_seed

labeled_data_ratio = args.label_ratio

importance = lambda len: "[" + ",".join([str(x) for x in [1] * len]) + "]"
feature=lambda features: "[" + ",".join([str(x) for x in features]) + "]"

save_dir_main = args.save_dir if args.save_dir else "abalation_study"
save_dir = f"{save_dir_main}/" \
           f"label_data_ration_{labeled_data_ratio}/" \
           f"feature_position_{'_'.join(args.feature_position)}/" \
           f"decoder_patch_{args.decoder_loss_padding_patchsize[1]}_padding_{args.decoder_loss_padding_patchsize[0]}/" \
           f"random_seed_{random_seed}"

common_opts = f" Data.labeled_data_ratio={args.label_ratio} " \
              f" Data.unlabeled_data_ratio={1 - args.label_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Trainer.max_epoch={args.max_epoch} " \
              f" Trainer.feature_names={feature(args.feature_position)}" \
              f" Trainer.feature_importance={importance(len(args.feature_position))}" \
              f" IICRegParameters.LossParams.paddings={args.decoder_loss_padding_patchsize[0]} " \
              f" IICRegParameters.LossParams.patch_sizes={args.decoder_loss_padding_patchsize[1]} "

jobs = [
    # repeat
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/cluster_10_subhead_10 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "
    f" IICRegParameters.EncoderParams.num_clusters=10 "
    f" IICRegParameters.EncoderParams.num_subheads=10 "
    f" IICRegParameters.DecoderParams.num_clusters=10 "
    f" IICRegParameters.DecoderParams.num_subheads=10 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/cluster_10_subhead_5 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "
    f" IICRegParameters.EncoderParams.num_clusters=10 "
    f" IICRegParameters.EncoderParams.num_subheads=5 "
    f" IICRegParameters.DecoderParams.num_clusters=10 "
    f" IICRegParameters.DecoderParams.num_subheads=5 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/cluster_10_subhead_2 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "
    f" IICRegParameters.EncoderParams.num_clusters=10 "
    f" IICRegParameters.EncoderParams.num_subheads=2 "
    f" IICRegParameters.DecoderParams.num_clusters=10 "
    f" IICRegParameters.DecoderParams.num_subheads=2 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/cluster_10_subhead_1 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "
    f" IICRegParameters.EncoderParams.num_clusters=10 "
    f" IICRegParameters.EncoderParams.num_subheads=1 "
    f" IICRegParameters.DecoderParams.num_clusters=10 "
    f" IICRegParameters.DecoderParams.num_subheads=1 ",

    # repeat
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/cluster_5_subhead_10 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "
    f" IICRegParameters.EncoderParams.num_clusters=5 "
    f" IICRegParameters.EncoderParams.num_subheads=10 "
    f" IICRegParameters.DecoderParams.num_clusters=5 "
    f" IICRegParameters.DecoderParams.num_subheads=10 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/cluster_5_subhead_5 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "
    f" IICRegParameters.EncoderParams.num_clusters=5 "
    f" IICRegParameters.EncoderParams.num_subheads=5 "
    f" IICRegParameters.DecoderParams.num_clusters=5 "
    f" IICRegParameters.DecoderParams.num_subheads=5 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/cluster_5_subhead_2 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "
    f" IICRegParameters.EncoderParams.num_clusters=5 "
    f" IICRegParameters.EncoderParams.num_subheads=2 "
    f" IICRegParameters.DecoderParams.num_clusters=5 "
    f" IICRegParameters.DecoderParams.num_subheads=2 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/cluster_5_subhead_1 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "
    f" IICRegParameters.EncoderParams.num_clusters=5 "
    f" IICRegParameters.EncoderParams.num_subheads=1 "
    f" IICRegParameters.DecoderParams.num_clusters=5 "
    f" IICRegParameters.DecoderParams.num_subheads=1 ",

    # repeat
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/cluster_20_subhead_10 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "
    f" IICRegParameters.EncoderParams.num_clusters=20 "
    f" IICRegParameters.EncoderParams.num_subheads=10 "
    f" IICRegParameters.DecoderParams.num_clusters=20 "
    f" IICRegParameters.DecoderParams.num_subheads=10 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/cluster_20_subhead_5 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "
    f" IICRegParameters.EncoderParams.num_clusters=20 "
    f" IICRegParameters.EncoderParams.num_subheads=5 "
    f" IICRegParameters.DecoderParams.num_clusters=20 "
    f" IICRegParameters.DecoderParams.num_subheads=5 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/cluster_20_subhead_2 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "
    f" IICRegParameters.EncoderParams.num_clusters=20 "
    f" IICRegParameters.EncoderParams.num_subheads=2 "
    f" IICRegParameters.DecoderParams.num_clusters=20 "
    f" IICRegParameters.DecoderParams.num_subheads=2 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/cluster_20_subhead_1 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "
    f" IICRegParameters.EncoderParams.num_clusters=20 "
    f" IICRegParameters.EncoderParams.num_subheads=1 "
    f" IICRegParameters.DecoderParams.num_clusters=20 "
    f" IICRegParameters.DecoderParams.num_subheads=1 ",

]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

jobsubmiter = JobSubmiter(project_path="./", on_local=True, time=args.time)
for j in jobs:
    jobsubmiter.prepare_env(["source ./venv/bin/activate ",
                             "export OMP_NUM_THREADS=1",
                             "export PYTHONOPTIMIZE=1"])
    jobsubmiter.account = next(accounts)
    jobsubmiter.run(j)

# from gpu_queue import JobSubmitter
#
# jobsubmiter = JobSubmitter(jobs, [0, 1])
# jobsubmiter.submit_jobs()
