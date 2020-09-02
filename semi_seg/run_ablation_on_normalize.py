import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", required=True, type=int, help="random seed")
seed = parser.parse_args().seed

labeled_data_ratio = 0.05
save_dir = f"abalation_normalization/seed_{seed}"
num_batches = 300
max_epoch = 100
time = 6

common_opts = f" Data.labeled_data_ratio={labeled_data_ratio} " \
              f" Data.unlabeled_data_ratio={1 - labeled_data_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "\
              f" Trainer.max_epoch={max_epoch} " \
              f" RandomSeed={seed} "
jobs = [
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/false_flase "
    f" IICRegParameters.EncoderParams.normalize=false "
    f" IICRegParameters.DecoderParams.normalize=false",
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/false_true "
    f" IICRegParameters.EncoderParams.normalize=false "
    f" IICRegParameters.DecoderParams.normalize=true",
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/true_flase "
    f" IICRegParameters.EncoderParams.normalize=true "
    f" IICRegParameters.DecoderParams.normalize=false",
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/true_true "
    f" IICRegParameters.EncoderParams.normalize=true "
    f" IICRegParameters.DecoderParams.normalize=true",

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
