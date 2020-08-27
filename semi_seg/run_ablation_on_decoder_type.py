from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

labeled_data_ratio = 0.05
save_dir = "0827/abalation_decoder_type"
num_batches = 300
max_epoch = 100
time = 6

importance_generator = lambda len: "[" + ",".join([str(x) for x in [1] * len]) + "]"
feature2string = lambda features: "[" + ",".join([str(x) for x in features]) + "]"

common_opts = f" Data.labeled_data_ratio={labeled_data_ratio} " \
              f" Data.unlabeled_data_ratio={1 - labeled_data_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Trainer.max_epoch={max_epoch} " \
              f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 "
jobs = [
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/mlp "
    f" IICRegParameters.DecoderParams.head_types=mlp ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/linear "
    f" IICRegParameters.DecoderParams.head_types=linear ",
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
