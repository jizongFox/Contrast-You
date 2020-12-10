import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", required=True, type=int, help="random seed")
seed = parser.parse_args().seed

labeled_data_ratio = 0.05
save_dir = f"1201/abalation_cluster_number/seed_{seed}"
num_batches = 300
max_epoch = 100
time = 4

common_opts = f" Data.labeled_data_ratio={labeled_data_ratio} " \
              f" Data.unlabeled_data_ratio={1 - labeled_data_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Trainer.max_epoch={max_epoch} " \
              f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 " \
              f" RandomSeed={seed} "

jobs = [
    # # encoder
    # f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder/cluster_num_10 "
    # f" IICRegParameters.EncoderParams.num_clusters=10   Trainer.feature_importance=[1,0,0] ",
    #
    # f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder/cluster_num_20 "
    # f" IICRegParameters.EncoderParams.num_clusters=20 Trainer.feature_importance=[1,0,0] ",
    #
    # f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder/cluster_num_5 "
    # f" IICRegParameters.EncoderParams.num_clusters=5 Trainer.feature_importance=[1,0,0] ",
    #
    # encoder decoder
    # f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/10_10 "
    # f" IICRegParameters.EncoderParams.num_clusters=10 IICRegParameters.DecoderParams.num_clusters=10 ",
    #
    # f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/10_5 "
    # f" IICRegParameters.EncoderParams.num_clusters=10 IICRegParameters.DecoderParams.num_clusters=5 ",
    #
    # f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/10_20 "
    # f" IICRegParameters.EncoderParams.num_clusters=10 IICRegParameters.DecoderParams.num_clusters=20 ",
    #
    # # encoder decoder
    # f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/5_10 "
    # f" IICRegParameters.EncoderParams.num_clusters=5 IICRegParameters.DecoderParams.num_clusters=10 ",
    #
    # f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/5_5 "
    # f" IICRegParameters.EncoderParams.num_clusters=5 IICRegParameters.DecoderParams.num_clusters=5 ",
    #
    # f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/5_20 "
    # f" IICRegParameters.EncoderParams.num_clusters=5 IICRegParameters.DecoderParams.num_clusters=20 ",
    #
    # # decoder
    # f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/20_10 "
    # f" IICRegParameters.EncoderParams.num_clusters=20 IICRegParameters.DecoderParams.num_clusters=10 ",
    #
    # f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/20_5 "
    # f" IICRegParameters.EncoderParams.num_clusters=20 IICRegParameters.DecoderParams.num_clusters=5 ",
    #
    # f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/20_20 "
    # f" IICRegParameters.EncoderParams.num_clusters=20 IICRegParameters.DecoderParams.num_clusters=20 ",

    #

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/2_2 "
    f" IICRegParameters.EncoderParams.num_clusters=2 IICRegParameters.DecoderParams.num_clusters=2 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/2_5 "
    f" IICRegParameters.EncoderParams.num_clusters=2 IICRegParameters.DecoderParams.num_clusters=5 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/2_10 "
    f" IICRegParameters.EncoderParams.num_clusters=2 IICRegParameters.DecoderParams.num_clusters=10 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/2_20 "
    f" IICRegParameters.EncoderParams.num_clusters=2 IICRegParameters.DecoderParams.num_clusters=20 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/5_2 "
    f" IICRegParameters.EncoderParams.num_clusters=5 IICRegParameters.DecoderParams.num_clusters=2 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/10_2 "
    f" IICRegParameters.EncoderParams.num_clusters=10 IICRegParameters.DecoderParams.num_clusters=2 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder_decoder/20_2 "
    f" IICRegParameters.EncoderParams.num_clusters=20 IICRegParameters.DecoderParams.num_clusters=2 ",
]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])
jobsubmiter = JobSubmiter(project_path="../", on_local=False, time=time, gres=f"gpu:1")
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

# from gpu_queue import JobSubmitter
#
# jobsubmiter = JobSubmitter(jobs, [0, 1])
# jobsubmiter.submit_jobs()
