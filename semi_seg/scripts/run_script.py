import argparse
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # noqa

parser.add_argument("-n", "--dataset_name", default="acdc", type=str)
parser.add_argument("-l", "--label_ratio", default=0.05, type=float)
parser.add_argument("-b", "--num_batches", default=500, type=int)
parser.add_argument("-e", "--max_epoch", default=100, type=int)
parser.add_argument("-s", "--random_seed", default=1, type=int)
parser.add_argument("--save_dir", default=None, type=str)
parser.add_argument("--time", default=4, type=int)
parser.add_argument("--distributed", action="store_true", default=False, help="enable distributed training")
parser.add_argument("--num_gpus", default=1, type=int, help="gpu numbers")

args = parser.parse_args()

assert (args.distributed and args.num_gpus > 1), (args.distributed, args.num_gpus)

num_batches = args.num_batches
random_seed = args.random_seed

labeled_data_ratio = args.label_ratio

dataset_name2class_numbers = {
    "acdc": 4,
    "prostate": 2,
    "spleen": 2,
}
lr_zooms = {"acdc": 0.0000001,
            "prostate": 0.000001,
            "spleen": 0.000001}

save_dir_main = args.save_dir if args.save_dir else "main_result_folder"
save_dir = f"{save_dir_main}/{args.dataset_name}/" \
           f"label_data_ration_{labeled_data_ratio}/" \
           f"random_seed_{random_seed}"

common_opts = f" Data.labeled_data_ratio={args.label_ratio} " \
              f" Data.unlabeled_data_ratio={1 - args.label_ratio} " \
              f" Trainer.num_batches={num_batches} " \
              f" Trainer.max_epoch={args.max_epoch} " \
              f" Data.name={args.dataset_name} " \
              f" Arch.num_classes={dataset_name2class_numbers[args.dataset_name]} " \
              f" Optim.lr={lr_zooms[args.dataset_name]:.10f} " \
              f" RandomSeed={random_seed} " \
              f" DistributedTrain={args.distributed}"

jobs = [
    f" python main.py {common_opts} Trainer.name=partial Trainer.save_dir={save_dir}/ps  ",

    f" python main.py {common_opts} Trainer.name=partial Trainer.save_dir={save_dir}/fs "
    f" Data.labeled_data_ratio=1 Data.unlabeled_data_ratio=0",
    # baseline
    # entropy
    f" python main.py {common_opts} Trainer.name=entropy Trainer.save_dir={save_dir}/entropy/0.00001 "
    f" EntropyMinParameters.weight=0.00001",

    f" python main.py {common_opts} Trainer.name=entropy Trainer.save_dir={save_dir}/entropy/0.0001 "
    f" EntropyMinParameters.weight=0.0001",

    f" python main.py {common_opts} Trainer.name=entropy Trainer.save_dir={save_dir}/entropy/0.001 "
    f" EntropyMinParameters.weight=0.001",
    # mean teacher
    f" python main.py {common_opts} Trainer.name=meanteacher Trainer.save_dir={save_dir}/mt/1 "
    f" MeanTeacherParameters.weight=1.0 ",

    f" python main.py {common_opts} Trainer.name=meanteacher Trainer.save_dir={save_dir}/mt/5 "
    f" MeanTeacherParameters.weight=5.0 ",

    f" python main.py {common_opts} Trainer.name=meanteacher Trainer.save_dir={save_dir}/mt/10 "
    f" MeanTeacherParameters.weight=10.0 ",

    f" python main.py {common_opts} Trainer.name=meanteacher Trainer.save_dir={save_dir}/mt/15 "
    f" MeanTeacherParameters.weight=15.0 ",

    # uda
    f" python main.py {common_opts} Trainer.name=uda Trainer.save_dir={save_dir}/uda/mse/1 "
    f" UDARegCriterion.name=mse UDARegCriterion.weight=1  ",

    f" python main.py {common_opts} Trainer.name=uda Trainer.save_dir={save_dir}/uda/mse/5 "
    f" UDARegCriterion.name=mse UDARegCriterion.weight=5  ",

    f" python main.py {common_opts} Trainer.name=uda Trainer.save_dir={save_dir}/uda/mse/10 "
    f" UDARegCriterion.name=mse UDARegCriterion.weight=10  ",

    f" python main.py {common_opts} Trainer.name=uda Trainer.save_dir={save_dir}/uda/mse/15 "
    f" UDARegCriterion.name=mse UDARegCriterion.weight=15  ",

    # iic

    f" python main.py {common_opts} Trainer.name=iic Trainer.save_dir={save_dir}/iic/0.001 "
    f" IICRegParameters.weight=0.001 ",

    f" python main.py {common_opts} Trainer.name=iic Trainer.save_dir={save_dir}/iic/0.01 "
    f" IICRegParameters.weight=0.01 ",

    f" python main.py {common_opts} Trainer.name=iic Trainer.save_dir={save_dir}/iic/0.1 "
    f" IICRegParameters.weight=0.1 ",

    # uda iic

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/1.0_0.1 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=1.0 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/1.0_0.01 "
    f" IICRegParameters.weight=0.01 UDARegCriterion.weight=1.0 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/1.0_0.05 "
    f" IICRegParameters.weight=0.05 UDARegCriterion.weight=1.0 ",

    # repeat
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/5.0_0.1 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=5.0 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/5.0_0.01 "
    f" IICRegParameters.weight=0.01 UDARegCriterion.weight=5.0 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/5.0_0.05 "
    f" IICRegParameters.weight=0.05 UDARegCriterion.weight=5.0 ",

    # repeat
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/10.0_0.1 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=10.0 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/10.0_0.01 "
    f" IICRegParameters.weight=0.01 UDARegCriterion.weight=10.0 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/10.0_0.05 "
    f" IICRegParameters.weight=0.05 UDARegCriterion.weight=10.0 ",

    # repeat
    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/15.0_0.1 "
    f" IICRegParameters.weight=0.1 UDARegCriterion.weight=15.0 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/15.0_0.01 "
    f" IICRegParameters.weight=0.01 UDARegCriterion.weight=15.0 ",

    f" python main.py {common_opts} Trainer.name=udaiic Trainer.save_dir={save_dir}/udaiic/15.0_0.05 "
    f" IICRegParameters.weight=0.05 UDARegCriterion.weight=15.0 ",

    # iicmeanteacher
    f" python main.py {common_opts} Trainer.name=iicmeanteacher Trainer.save_dir={save_dir}/iicmeanteacher/1.0_0.1 "
    f" IICRegParameters.weight=0.1 MeanTeacherParameters.weight=1.0 ",

    f" python main.py {common_opts} Trainer.name=iicmeanteacher Trainer.save_dir={save_dir}/iicmeanteacher/1.0_0.01 "
    f" IICRegParameters.weight=0.01 MeanTeacherParameters.weight=1.0 ",

    f" python main.py {common_opts} Trainer.name=iicmeanteacher Trainer.save_dir={save_dir}/iicmeanteacher/1.0_0.05 "
    f" IICRegParameters.weight=0.05 MeanTeacherParameters.weight=1.0 ",

    # repeat
    f" python main.py {common_opts} Trainer.name=iicmeanteacher Trainer.save_dir={save_dir}/iicmeanteacher/5.0_0.1 "
    f" IICRegParameters.weight=0.1 MeanTeacherParameters.weight=5.0 ",

    f" python main.py {common_opts} Trainer.name=iicmeanteacher Trainer.save_dir={save_dir}/iicmeanteacher/5.0_0.01 "
    f" IICRegParameters.weight=0.01 MeanTeacherParameters.weight=5.0 ",

    f" python main.py {common_opts} Trainer.name=iicmeanteacher Trainer.save_dir={save_dir}/iicmeanteacher/5.0_0.05 "
    f" IICRegParameters.weight=0.05 MeanTeacherParameters.weight=5.0 ",

    # repeat
    f" python main.py {common_opts} Trainer.name=iicmeanteacher Trainer.save_dir={save_dir}/iicmeanteacher/10.0_0.1 "
    f" IICRegParameters.weight=0.1 MeanTeacherParameters.weight=10.0 ",

    f" python main.py {common_opts} Trainer.name=iicmeanteacher Trainer.save_dir={save_dir}/iicmeanteacher/10.0_0.01 "
    f" IICRegParameters.weight=0.01 MeanTeacherParameters.weight=10.0 ",

    f" python main.py {common_opts} Trainer.name=iicmeanteacher Trainer.save_dir={save_dir}/iicmeanteacher/10.0_0.05 "
    f" IICRegParameters.weight=0.05 MeanTeacherParameters.weight=10.0 ",

    # repeat
    f" python main.py {common_opts} Trainer.name=iicmeanteacher Trainer.save_dir={save_dir}/iicmeanteacher/15.0_0.1 "
    f" IICRegParameters.weight=0.1 MeanTeacherParameters.weight=15.0 ",

    f" python main.py {common_opts} Trainer.name=iicmeanteacher Trainer.save_dir={save_dir}/iicmeanteacher/15.0_0.01 "
    f" IICRegParameters.weight=0.01 MeanTeacherParameters.weight=15.0 ",

    f" python main.py {common_opts} Trainer.name=iicmeanteacher Trainer.save_dir={save_dir}/iicmeanteacher/15.0_0.05 "
    f" IICRegParameters.weight=0.05 MeanTeacherParameters.weight=15.0 ",

    # midl
    f" python main.py {common_opts} Trainer.name=midl Trainer.save_dir={save_dir}/midl/10_0.1 "
    f" MIDLPaperParameters.iic_weight=0.1  UDARegCriterion.weight=10",

    f" python main.py {common_opts} Trainer.name=midl Trainer.save_dir={save_dir}/midl/10_0.5 "
    f" MIDLPaperParameters.iic_weight=0.5  UDARegCriterion.weight=10",

    f" python main.py {common_opts} Trainer.name=midl Trainer.save_dir={save_dir}/midl/10_0.01 "
    f" MIDLPaperParameters.iic_weight=0.01  UDARegCriterion.weight=10",

    f" python main.py {common_opts} Trainer.name=midl Trainer.save_dir={save_dir}/midl/5_0.1 "
    f" MIDLPaperParameters.iic_weight=0.1  UDARegCriterion.weight=5",

    f" python main.py {common_opts} Trainer.name=midl Trainer.save_dir={save_dir}/midl/5_0.5 "
    f" MIDLPaperParameters.iic_weight=0.5  UDARegCriterion.weight=5",

    f" python main.py {common_opts} Trainer.name=midl Trainer.save_dir={save_dir}/midl/5_0.01 "
    f" MIDLPaperParameters.iic_weight=0.01  UDARegCriterion.weight=5",

    f" python main.py {common_opts} Trainer.name=midl Trainer.save_dir={save_dir}/midl/15_0.1 "
    f" MIDLPaperParameters.iic_weight=0.1  UDARegCriterion.weight=15",

    f" python main.py {common_opts} Trainer.name=midl Trainer.save_dir={save_dir}/midl/15_0.5 "
    f" MIDLPaperParameters.iic_weight=0.5  UDARegCriterion.weight=15",

    f" python main.py {common_opts} Trainer.name=midl Trainer.save_dir={save_dir}/midl/15_0.01 "
    f" MIDLPaperParameters.iic_weight=0.01  UDARegCriterion.weight=15",

    # pica

    f" python main.py {common_opts} Trainer.name=pica Trainer.save_dir={save_dir}/pica/0.001 "
    f" PICARegParameters.weight=0.001 Trainer.feature_names=[Conv5,] Trainer.feature_importance=[1,] ",

    f" python main.py {common_opts} Trainer.name=pica Trainer.save_dir={save_dir}/pica/0.01 "
    f" PICARegParameters.weight=0.01 Trainer.feature_names=[Conv5,] Trainer.feature_importance=[1,] ",

    f" python main.py {common_opts} Trainer.name=pica Trainer.save_dir={save_dir}/pica/0.1 "
    f" PICARegParameters.weight=0.1 Trainer.feature_names=[Conv5,] Trainer.feature_importance=[1,] ",

    f" python main.py {common_opts} Trainer.name=iic Trainer.save_dir={save_dir}/iic/0.001 "
    f" IICRegParameters.weight=0.001 Trainer.feature_names=[Conv5,] Trainer.feature_importance=[1,] ",

    f" python main.py {common_opts} Trainer.name=iic Trainer.save_dir={save_dir}/iic/0.01 "
    f" IICRegParameters.weight=0.01 Trainer.feature_names=[Conv5,] Trainer.feature_importance=[1,] ",

    f" python main.py {common_opts} Trainer.name=iic Trainer.save_dir={save_dir}/iic/0.1 "
    f" IICRegParameters.weight=0.1 Trainer.feature_names=[Conv5,] Trainer.feature_importance=[1,]",

]

# CC things
accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])

jobsubmiter = JobSubmiter(project_path="../", on_local=False, time=args.time, gres=f"gpu:{args.num_gpus}")
for j in jobs:
    jobsubmiter.prepare_env(
        [
            "source ../venv/bin/activate ",
            "export OMP_NUM_THREADS=1",
            "export PYTHONOPTIMIZE=1"
        ]
    )
    jobsubmiter.account = next(accounts)
    jobsubmiter.run(j)

# from gpu_queue import JobSubmitter
#
# jobsubmiter = JobSubmitter(jobs, [0, 1])
# jobsubmiter.submit_jobs()
