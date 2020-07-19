from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

save_dir = "first_try"
num_batches = 2000
random_seed = 1

labeled_data_ratio = 0.05
unlabeled_data_ratio = 1 - labeled_data_ratio

common_opts = f" RandomSeed={random_seed} Data.labeled_data_ratio={labeled_data_ratio} Data.unlabeled_data_ratio={unlabeled_data_ratio} Trainer.num_batches={num_batches} "
jobs = [
    f"python main_contrast.py {common_opts} Trainer.save_dir={save_dir}/baseline  Trainer.train_encoder=False Trainer.train_decoder=False ",
    f"python main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder  Trainer.train_encoder=True Trainer.train_decoder=False ",
    f"python main_contrast.py {common_opts} Trainer.save_dir={save_dir}/encoder_decoder Trainer.train_encoder=True Trainer.train_decoder=True "
]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

jobsubmiter = JobSubmiter(project_path="./", on_local=False)
for j in jobs:
    jobsubmiter.prepare_env(["export OMP_NUM_THREADS=1", "source "])
    jobsubmiter.account = next(accounts)
    jobsubmiter.run(j)
