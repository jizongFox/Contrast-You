import itertools
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

labeled_data_ratio = 0.05
save_dir = "0827/abalation_cluster_number"
num_batches = 300
max_epoch = 100
time = 6

common_opts = f" Data.labeled_data_ratio={labeled_data_ratio} " \
              f" Data.unlabeled_data_ratio={1 - labeled_data_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Trainer.max_epoch={max_epoch} " \
              f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "

jobs = [
    # encoder
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder/cluster_num_10 "
    f" IICRegParameters.EncoderParams.num_clusters=10 "
    
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder/cluster_num_20 "
    f" IICRegParameters.EncoderParams.num_clusters=20 "
    
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/encoder/cluster_num_5 "
    f" IICRegParameters.EncoderParams.num_clusters=5 "


]


# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

jobsubmiter = JobSubmiter(project_path="./", on_local=False, time=time)

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
