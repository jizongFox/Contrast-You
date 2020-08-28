import argparse
import itertools
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", required=True, type=int, help="random seed")
seed = parser.parse_args().seed

labeled_data_ratio = 0.05
save_dir = f"0828/abalation_feature/seed_{seed}"
num_batches = 300
max_epoch = 100
time = 6

importance_generator = lambda len: "[" + ",".join([str(x) for x in [1] * len]) + "]"
feature2string = lambda features: "[" + ",".join([str(x) for x in features]) + "]"

common_opts = f" Data.labeled_data_ratio={labeled_data_ratio} " \
              f" Data.unlabeled_data_ratio={1 - labeled_data_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Trainer.max_epoch={max_epoch} " \
              f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 " \
              f" RandomSeed={seed} "

jobs_generator = lambda features: [
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/{'_'.join(features)} "
    f" Trainer.feature_names={feature2string(features)}"
    f" Trainer.feature_importance={importance_generator(len(features))} "
][0]


def feature_generator(features, r=1):
    import itertools
    result_set = set(itertools.combinations(features, r))
    for r in result_set:
        yield list(r)


features = [
    "Conv5", "Up_conv5", "Up_conv4", "Up_conv3", "Up_conv2"
]

jobs = []
for f in itertools.chain.from_iterable(
    [feature_generator(features, r=3),
     feature_generator(features, r=2),
     feature_generator(features, r=1)]
):
    jobs.append(jobs_generator(f))

# print(jobs)

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
