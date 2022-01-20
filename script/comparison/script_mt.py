import argparse
import os
from itertools import cycle
from typing import Sequence, List, Iterator, Optional

from loguru import logger

from contrastyou import __accounts, on_cc, OPT_PATH, git_hash
from contrastyou.configure import yaml_load
from contrastyou.submitter import SlurmSubmitter
from script.comparison.script_ent import enable_acdc_all_class_train
from script.script_generator_pretrain_cc import get_hyper_param_string, run_baseline
from script.utils import grid_search, move_dataset


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("save_dir", type=str, help="save dir")
    parser.add_argument(
        "--data-name", type=str,
        choices=("acdc", "acdc_lv", "acdc_rv", "acdc_myo", "prostate", "spleen", "hippocampus"),
        default="acdc",
        help="dataset_choice"
    )
    parser.add_argument("--max-epoch", default=30, type=int, help="max epoch")
    parser.add_argument("--num-batches", default=300, type=int, help="number of batches")
    parser.add_argument("--seeds", type=int, nargs="+", default=(10,), )
    parser.add_argument("--force-show", action="store_true", help="showing script")
    args = parser.parse_args()
    return args


def _run_semi(*, save_dir: str, random_seed: int = 10, num_labeled_scan: int, max_epoch: int,
              num_batches: int, arch_checkpoint: str, lr: float, data_name: str = "acdc",
              mt_weight: float, hard_clip: str
              ):
    return f""" python main_nd.py RandomSeed={random_seed} Trainer.name=semi \
     Trainer.save_dir={save_dir} Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} Data.name={data_name} \
    Data.labeled_scan_num={num_labeled_scan}  Arch.checkpoint={arch_checkpoint} Optim.lr={lr:.10f} \
    MeanTeacherParameters.weight={mt_weight:.10f} \
    MeanTeacherParameters.hard_clip={hard_clip}  \
    --path   config/base.yaml  config/hooks/mt.yaml  \
    """


def _run_semi_per_class(*, save_dir: str, random_seed: int = 10, num_labeled_scan: int, max_epoch: int,
                        num_batches: int, arch_checkpoint: str, lr: float, data_name: str = "acdc",
                        mt_weight: float, hard_clip: str
                        ):
    assert data_name == "acdc", data_name
    return f""" python main_nd.py RandomSeed={random_seed} Trainer.name=semi \
     Trainer.save_dir={save_dir}/lv Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} Data.name={data_name}_lv \
    Data.labeled_scan_num={num_labeled_scan}  Arch.checkpoint={arch_checkpoint} Optim.lr={lr:.10f} \
    MeanTeacherParameters.weight={mt_weight:.10f} \
    MeanTeacherParameters.hard_clip={hard_clip}  \
    --path   config/base.yaml  config/hooks/mt.yaml  \
    &&\
    python main_nd.py RandomSeed={random_seed} Trainer.name=semi \
     Trainer.save_dir={save_dir}/rv Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} Data.name={data_name}_rv \
    Data.labeled_scan_num={num_labeled_scan}  Arch.checkpoint={arch_checkpoint} Optim.lr={lr:.10f} \
    MeanTeacherParameters.weight={mt_weight:.10f} \
    MeanTeacherParameters.hard_clip={hard_clip}  \
    --path   config/base.yaml  config/hooks/mt.yaml  \
    &&\
    python main_nd.py RandomSeed={random_seed} Trainer.name=semi \
     Trainer.save_dir={save_dir}/myo Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} Data.name={data_name}_myo \
    Data.labeled_scan_num={num_labeled_scan}  Arch.checkpoint={arch_checkpoint} Optim.lr={lr:.10f} \
    MeanTeacherParameters.weight={mt_weight:.10f} \
    MeanTeacherParameters.hard_clip={hard_clip}  \
    --path   config/base.yaml  config/hooks/mt.yaml \
    """


def run_semi_regularize(
        *, save_dir, random_seed: int = 10, max_epoch: int, num_batches: int, data_name: str = "acdc",
        ent_weight: float, hard_clip: str
) -> List[str]:
    data_opt = yaml_load(os.path.join(OPT_PATH, data_name + ".yaml"))
    labeled_scans = data_opt["labeled_ratios"][:-1]
    if data_name == "acdc" and not enable_acdc_all_class_train:
        run_semi = _run_semi_per_class
    else:
        run_semi = _run_semi

    semi_script = [
        run_semi(
            save_dir=os.path.join(save_dir, "semi", f"labeled_num_{l:03d}"), random_seed=random_seed,
            num_labeled_scan=l, max_epoch=max_epoch, num_batches=num_batches, arch_checkpoint="null",
            lr=data_opt["ft_lr"], data_name=data_name,
            mt_weight=ent_weight, hard_clip=hard_clip,
        )
        for l in labeled_scans
    ]
    return semi_script


def run_baseline_with_grid_search(*, save_dir, random_seeds: Sequence[int] = 10, max_epoch: int, num_batches: int,
                                  data_name: str = "acdc"):
    rand_seed_gen = grid_search(random_seed=random_seeds)
    for random_seed in rand_seed_gen:
        yield run_baseline(save_dir=os.path.join(save_dir, f"seed_{random_seed['random_seed']}"),
                           **random_seed, max_epoch=max_epoch, num_batches=num_batches,
                           data_name=data_name)


def run_semi_regularize_with_grid_search(
        *, save_dir, random_seeds: Sequence[int] = 10, max_epoch: int, num_batches: int,
        data_name: str, mt_weight: Sequence[float], hard_clip: Sequence[str],
        max_num: Optional[int] = 200,
) -> Iterator[List[str]]:
    param_generator = grid_search(
        random_seed=random_seeds, max_num=max_num, mt_weight=mt_weight, hard_clip=hard_clip,
    )
    for param in param_generator:
        random_seed = param.pop("random_seed")
        sp_str = get_hyper_param_string(**param)
        yield run_semi_regularize(save_dir=os.path.join(save_dir, f"seed_{random_seed}", sp_str),
                                  random_seed=random_seed,
                                  max_epoch=max_epoch, num_batches=num_batches, data_name=data_name, **param)


if __name__ == '__main__':
    args = get_args()
    account = cycle(__accounts)
    on_local = not on_cc()
    force_show = args.force_show
    data_name = args.data_name
    random_seeds = args.seeds
    max_epoch = args.max_epoch
    num_batches = args.num_batches
    save_dir = args.save_dir

    save_dir = os.path.join(save_dir, f"hash_{git_hash}/{data_name}")

    submitter = SlurmSubmitter(work_dir="../../", stop_on_error=on_local, on_local=on_local)
    submitter.configure_environment([
        # "set -e "
        "module load python/3.8.2 ",
        f"source ~/venv/bin/activate ",
        'if [ $(which python) == "/usr/bin/python" ]',
        "then",
        "exit 9",
        "fi",
        "export OMP_NUM_THREADS=1",
        "export PYTHONOPTIMIZE=1",
        "export PYTHONWARNINGS=ignore ",
        "export CUBLAS_WORKSPACE_CONFIG=:16:8 ",
        "export LOGURU_LEVEL=TRACE",
        "echo $(pwd)",
        move_dataset(),
        "nvidia-smi",
        "python -c 'import torch; print(torch.randn(1,1,1,1,device=\"cuda\"))'"
    ])
    submitter.configure_sbatch(mem=24)

    # baseline
    job_generator = run_baseline_with_grid_search(
        save_dir=os.path.join(save_dir, "mt"), random_seeds=random_seeds, max_epoch=max_epoch,
        num_batches=num_batches, data_name=data_name)

    jobs = list(job_generator)
    logger.info(f"logging {len(jobs)} jobs")
    for job in jobs:
        submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))

    # only with RR on semi supervised case
    job_generator = run_semi_regularize_with_grid_search(save_dir=os.path.join(save_dir, "mt"),
                                                         random_seeds=random_seeds,
                                                         max_epoch=max_epoch, num_batches=num_batches,
                                                         data_name=data_name,
                                                         mt_weight=(0, 0.001, 0.01, 0.1, 1, 10),
                                                         hard_clip=("true", "false")
                                                         )

    jobs = list(job_generator)
    logger.info(f"logging {len(jobs)} jobs")
    for job in jobs:
        submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))
