import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--label_ratio", default=0.05, type=float)
parser.add_argument("-b", "--num_batches", default=500, type=int)
parser.add_argument("-e", "--max_epoch", default=100, type=int)
parser.add_argument("-s", "--random_seed", default=1, type=int)
parser.add_argument("--save_dir", default=None, type=str)
parser.add_argument("--time", default=4, type=int)

args = parser.parse_args()

num_batches = args.num_batches
random_seed = args.random_seed

labeled_data_ratio = args.label_ratio

save_dir_main = args.save_dir if args.save_dir else "main_result_folder"
save_dir = f"{save_dir_main}/" \
           f"label_data_ration_{labeled_data_ratio}"

common_opts = f" Data.labeled_data_ratio={args.label_ratio} " \
              f" Data.unlabeled_data_ratio={1 - args.label_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Trainer.max_epoch={args.max_epoch} "

jobs = [
    f" python main.py {common_opts} Trainer.name=partial Trainer.save_name={save_dir}/ps  ",

    f" python main.py {common_opts} Trainer.name=FS Trainer.save_name={save_dir}/fs "
    f" Data.labeled_data_ratio=1 Data.unlabeled_data_ratio=0",

    f" python main.py {common_opts} Trainer.name=uda Trainer.save_name={save_dir}/uda/mse/0.1 "
    f" UDARegCriterion.name=mse UDARegCriterion.weight=0.1  ",

    f" python main.py {common_opts} Trainer.name=uda Trainer.save_name={save_dir}/uda/mse/1 "
    f" UDARegCriterion.name=mse UDARegCriterion.weight=1  ",

    f" python main.py {common_opts} Trainer.name=uda Trainer.save_name={save_dir}/uda/mse/5 "
    f" UDARegCriterion.name=mse UDARegCriterion.weight=5  ",

    f" python main.py {common_opts} Trainer.name=iic Trainer.save_name={save_dir}/uda/iic/0.001 "
    f" IICRegParameters.LocalCluster.num_subheads=8  IICRegParameters.LocalCluster.num_clusters=10 "
    f" IICRegParameters.weight=0.001 ",

    f" python main.py {common_opts} Trainer.name=iic Trainer.save_name={save_dir}/uda/iic/0.01 "
    f" IICRegParameters.LocalCluster.num_subheads=8  IICRegParameters.LocalCluster.num_clusters=10 "
    f" IICRegParameters.weight=0.01 ",

    f" python main.py {common_opts} Trainer.name=iic Trainer.save_name={save_dir}/uda/iic/0.1 "
    f" IICRegParameters.LocalCluster.num_subheads=8  IICRegParameters.LocalCluster.num_clusters=10 "
    f" IICRegParameters.weight=0.1 ",
]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

jobsubmiter = JobSubmiter(project_path="./", on_local=False, time=args.time)
for j in jobs:
    jobsubmiter.prepare_env(["source ./venv/bin/activate ", "export OMP_NUM_THREADS=1", ])
    jobsubmiter.account = next(accounts)
    jobsubmiter.run(j)

from gpu_queue import JobSubmitter

jobsubmiter = JobSubmitter(jobs, [0, 1])
jobsubmiter.submit_jobs()
