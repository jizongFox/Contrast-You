import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--label_ratio", default=0.05, type=float)
parser.add_argument("-n", "--trainer_name", required=True, type=str)
parser.add_argument("-b", "--num_batches", default=100, type=int)
parser.add_argument("-s", "--random_seed", default=1, type=int)
parser.add_argument("-o", "--contrast_on", default="partition", type=str)
parser.add_argument("-c", "--num_clusters", default=40, type=int)

args = parser.parse_args()

num_batches = args.num_batches
random_seed = args.random_seed

labeled_data_ratio = args.label_ratio
unlabeled_data_ratio = 1 - labeled_data_ratio

trainer_name = args.trainer_name
assert trainer_name == "iiccontrast"
contrast_on = args.contrast_on
save_dir = f"iic_contrast3/label_data_ration_{labeled_data_ratio}/{trainer_name}/contrast_on_{contrast_on}"

common_opts = f" Trainer.name={trainer_name} PretrainEncoder.group_option={contrast_on} " \
              f" PretrainEncoder.num_clusters={args.num_clusters} RandomSeed={random_seed} " \
              f" Data.labeled_data_ratio={labeled_data_ratio} Data.unlabeled_data_ratio={unlabeled_data_ratio} " \
              f" Trainer.num_batches={num_batches} "
if trainer_name == "contrastMT":
    common_opts += f" FineTune.reg_weight={args.reg_weight} "

jobs = [
    f"python -O main_contrast.py {common_opts} Trainer.save_dir={save_dir}/without_pretrain  Trainer.train_encoder=False Trainer.train_decoder=False ",
    f"python -O main_contrast.py {common_opts} Trainer.save_dir={save_dir}/contrast_iic_0.0  Trainer.train_encoder=True Trainer.train_decoder=False PretrainEncoder.iic_weight=0.0",
    f"python -O main_contrast.py {common_opts} Trainer.save_dir={save_dir}/contrast_iic_1.0  Trainer.train_encoder=True Trainer.train_decoder=False PretrainEncoder.iic_weight=1.0",
    f"python -O main_contrast.py {common_opts} Trainer.save_dir={save_dir}/contrast_iic_3.0  Trainer.train_encoder=True Trainer.train_decoder=False PretrainEncoder.iic_weight=3.0",
    f"python -O main_contrast.py {common_opts} Trainer.save_dir={save_dir}/contrast_iic_5.0  Trainer.train_encoder=True Trainer.train_decoder=False PretrainEncoder.iic_weight=5.0",
    f"python -O main_contrast.py {common_opts} Trainer.save_dir={save_dir}/contrast_iic_10.0  Trainer.train_encoder=True Trainer.train_decoder=False PretrainEncoder.iic_weight=10.0",
    f"python -O main_contrast.py {common_opts} Trainer.save_dir={save_dir}/only_iic  Trainer.train_encoder=True Trainer.train_decoder=False PretrainEncoder.iic_weight=1 "
                                            f" PretrainEncoder.disable_contrastive=True ",
]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

jobsubmiter = JobSubmiter(project_path="./", on_local=False, time=4)
for j in jobs:
    jobsubmiter.prepare_env(["source ./venv/bin/activate ", "export OMP_NUM_THREADS=1", ])
    jobsubmiter.account = next(accounts)
    jobsubmiter.run(j)
