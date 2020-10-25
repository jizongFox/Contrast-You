import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", required=True, type=int, help="random seed")
seed = parser.parse_args().seed

labeled_data_ratio = 0.05
save_dir = f"0828/abalation_decoder_type/seed_{seed}"
num_batches = 300
max_epoch = 100
time = 6

common_opts = f" Data.labeled_data_ratio={labeled_data_ratio} " \
              f" Data.unlabeled_data_ratio={1 - labeled_data_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Trainer.max_epoch={max_epoch} " \
              f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 " \
              f" RandomSeed={seed} "
jobs = [
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/mlp_mlp "
    f" IICRegParameters.EncoderParams.head_types=mlp IICRegParameters.DecoderParams.head_types=mlp ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/linear_linear "
    f" IICRegParameters.EncoderParams.head_types=linear IICRegParameters.DecoderParams.head_types=linear ",


    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/mlp_linear "
    f" IICRegParameters.EncoderParams.head_types=mlp IICRegParameters.DecoderParams.head_types=linear ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/linear_mlp "
    f" IICRegParameters.EncoderParams.head_types=linear IICRegParameters.DecoderParams.head_types=mlp ",
]

jobsubmiter = JobSubmiter(project_path="./", on_local=False, time=time)
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

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