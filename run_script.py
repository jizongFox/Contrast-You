import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--label_ratio", default=0.05, type=float)
parser.add_argument("-n", "--trainer_name", required=True, type=str)
parser.add_argument("-b", "--num_batches", default=500, type=int)
parser.add_argument("-s", "--random_seed", default=1, type=int)
parser.add_argument("-o", "--contrast_on", default="partition", type=str)
parser.add_argument("-w", "--reg_weight", default=0.0, type=float)
parser.add_argument("--save_dir", default=None, type=str)
parser.add_argument("-a", "--augment", default="simple", type=str)
parser.add_argument("-g", "--group_sample_num", default=6, type=int)
parser.add_argument("--time", default=6, type=int)


args = parser.parse_args()

num_batches = args.num_batches
random_seed = args.random_seed

labeled_data_ratio = args.label_ratio

# trainer_name="contrast" # or contrastMT
trainer_name = args.trainer_name
contrast_on = args.contrast_on
save_dir_main = args.save_dir if args.save_dir else "0801_main_contrast"
save_dir = f"{save_dir_main}/" \
           f"label_data_ration_{labeled_data_ratio}/" \
           f"{trainer_name}/" \
           f"contrast_on_{contrast_on}/" \
           f"augment_{args.augment}/" \
           f"group_sample_{args.group_sample_num}"

if trainer_name == "contrastMT":
    save_dir = save_dir + f"/reg_weight_{args.reg_weight:.2f}"

common_opts = f" Trainer.name={trainer_name} " \
              f" PretrainEncoder.group_option={contrast_on} " \
              f" RandomSeed={random_seed} " \
              f" Data.labeled_data_ratio={labeled_data_ratio} " \
              f" Data.unlabeled_data_ratio={1.0 - labeled_data_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Augment={args.augment} " \
              f" ContrastData.group_sample_num={args.group_sample_num}"

if trainer_name == "contrastMT":
    common_opts += f" FineTune.reg_weight={args.reg_weight} "

jobs = [
    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/baseline  Trainer.train_encoder=False Trainer.train_decoder=False ",
    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder  Trainer.train_encoder=True Trainer.train_decoder=False ",
    f"python -O  main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder_decoder Trainer.train_encoder=True Trainer.train_decoder=True "
]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

jobsubmiter = JobSubmiter(project_path="./", on_local=False, time=args.time)
for j in jobs:
    jobsubmiter.prepare_env(["source ./venv/bin/activate ", "export OMP_NUM_THREADS=1", ])
    jobsubmiter.account = next(accounts)
    jobsubmiter.run(j)
