import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", required=True, type=int, help="random seed")
seed = parser.parse_args().seed

labeled_data_ratio = 0.05
save_dir = f"0828/abalation_padding_size/seed_{seed}"
num_batches = 300
max_epoch = 100
time = 12

common_opts = f" Data.labeled_data_ratio={labeled_data_ratio} " \
              f" Data.unlabeled_data_ratio={1 - labeled_data_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Trainer.max_epoch={max_epoch} " \
              f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 " \
              f" RandomSeed={seed} "

jobs = [
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[0.0,0.0]/padding_1024/ "
    f" LossParams.paddings=[0.0,0.0] "
    f" LossParams.patch_sizes=1024 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[0.0,1.0]/padding_1024/ "
    f" LossParams.paddings=[0.0,1.0] "
    f" LossParams.patch_sizes=1024 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[0.0,3.0]/padding_1024/ "
    f" LossParams.paddings=[0.0,3.0] "
    f" LossParams.patch_sizes=1024 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[0.0,5.0]/padding_1024/ "
    f" LossParams.paddings=[0.0,5.0] "
    f" LossParams.patch_sizes=1024 ",

    # repeat
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[1.0,0.0]/padding_1024/ "
    f" LossParams.paddings=[1.0,0.0] "
    f" LossParams.patch_sizes=1024 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[1.0,1.0]/padding_1024/ "
    f" LossParams.paddings=[1.0,1.0] "
    f" LossParams.patch_sizes=1024 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[1.0,3.0]/padding_1024/ "
    f" LossParams.paddings=[1.0,3.0] "
    f" LossParams.patch_sizes=1024 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[1.0,5.0]/padding_1024/ "
    f" LossParams.paddings=[1.0,5.0] "
    f" LossParams.patch_sizes=1024 ",

    # repeat
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[3.0,0.0]/padding_1024/ "
    f" LossParams.paddings=[3.0,0.0] "
    f" LossParams.patch_sizes=1024 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[3.0,1.0]/padding_1024/ "
    f" LossParams.paddings=[3.0,1.0] "
    f" LossParams.patch_sizes=1024 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[3.0,3.0]/padding_1024/ "
    f" LossParams.paddings=[3.0,3.0] "
    f" LossParams.patch_sizes=1024 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[3.0,5.0]/padding_1024/ "
    f" LossParams.paddings=[3.0,5.0] "
    f" LossParams.patch_sizes=1024 ",

    # repeat
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[5.0,0.0]/padding_1024/ "
    f" LossParams.paddings=[5.0,0.0] "
    f" LossParams.patch_sizes=1024 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[5.0,1.0]/padding_1024/ "
    f" LossParams.paddings=[5.0,1.0] "
    f" LossParams.patch_sizes=1024 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[5.0,3.0]/padding_1024/ "
    f" LossParams.paddings=[5.0,3.0] "
    f" LossParams.patch_sizes=1024 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/padding_[5.0,5.0]/padding_1024/ "
    f" LossParams.paddings=[5.0,5.0] "
    f" LossParams.patch_sizes=1024 ",

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
