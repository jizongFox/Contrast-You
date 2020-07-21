import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

parser = argparse.ArgumentParser()

parser.add_argument("--label_ratio", default=0.1, type=float)
parser.add_argument("--trainer_name", required=True, type=str)
parser.add_argument("--num_batches", default=500, type=int)
parser.add_argument("--random_seed", default=1, type=int)
args = parser.parse_args()

num_batches = args.num_batches
random_seed = args.random_seed

labeled_data_ratio = args.label_ratio
unlabeled_data_ratio = 1 - labeled_data_ratio

# trainer_name="contrast" # or contrastMT
trainer_name = args.trainer_name
save_dir = f"label_data_ration_{labeled_data_ratio}/{trainer_name}"

common_opts = f" Trainer.name={trainer_name} RandomSeed={random_seed} Data.labeled_data_ratio={labeled_data_ratio} Data.unlabeled_data_ratio={unlabeled_data_ratio} Trainer.num_batches={num_batches} "
jobs = [
    f"python main_contrast.py {common_opts} Trainer.save_dir={save_dir}/baseline  Trainer.train_encoder=False Trainer.train_decoder=False ",
    f"python main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder  Trainer.train_encoder=True Trainer.train_decoder=False ",
    f"python main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder_decoder Trainer.train_encoder=True Trainer.train_decoder=True "
]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

jobsubmiter = JobSubmiter(project_path="./", on_local=False, time=8)
for j in jobs:
    jobsubmiter.prepare_env(["source ./venv/bin/activate ", "export OMP_NUM_THREADS=1", ])
    jobsubmiter.account = next(accounts)
    jobsubmiter.run(j)
