import argparse
import os
from itertools import cycle
from typing import Sequence, List, Iterator, Optional

from loguru import logger

from contrastyou import __accounts, on_cc, MODEL_PATH, OPT_PATH, git_hash
from contrastyou.configure import yaml_load
from contrastyou.submitter import SlurmSubmitter
from script.script_generator_pretrain_cc import _run_ft, _run_ft_per_class, get_hyper_param_string, \
    run_baseline_with_grid_search, enable_acdc_all_class_train
from script.utils import grid_search, move_dataset


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("save_dir", type=str, help="save dir")
    parser.add_argument("--data-name", type=str, choices=("acdc", "acdc_lv", "acdc_rv", "acdc_myo", "prostate"),
                        default="acdc",
                        help="dataset_choice")
    parser.add_argument("--max-epoch-pretrain", default=50, type=int, help="max epoch")
    parser.add_argument("--max-epoch", default=30, type=int, help="max epoch")
    parser.add_argument("--num-batches", default=300, type=int, help="number of batches")
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, ], )
    parser.add_argument("--pretrain-scan-num", type=int, default=6, help="default `scan_sample_num` for pretraining")
    parser.add_argument("--force-show", action="store_true", help="showing script")
    args = parser.parse_args()
    return args


def _run_pretrain_cc(*, save_dir: str, random_seed: int = 10, max_epoch: int, num_batches: int, lr: float,
                     scan_sample_num: int, data_name: str = "acdc", imsat_weight: float, cons_weight: float,
                     num_clusters: int, num_subheads: int, head_type: str, imsat_lamda: float, use_dynamic: str):
    return f"""  python main_nd.py RandomSeed={random_seed} Trainer.name=pretrain_decoder Trainer.save_dir={save_dir} \
    Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches}  Optim.lr={lr:.10f} Data.name={data_name} \
    CrossCorrelationParameters.num_clusters={num_clusters}  \
    CrossCorrelationParameters.num_subheads={num_subheads}  \
    CrossCorrelationParameters.head_type={head_type}  \
    CrossCorrelationParameters.hooks.imsat.weight={imsat_weight:.10f} \
    CrossCorrelationParameters.hooks.imsat.lamda={imsat_lamda:.10f} \
    CrossCorrelationParameters.hooks.imsat.use_dynamic={use_dynamic} \
    CrossCorrelationParameters.hooks.consist.weight={cons_weight:.10f} \
    ContrastiveLoaderParams.scan_sample_num={scan_sample_num}  \
    --path config/base.yaml config/pretrain.yaml config/hooks/ccblocks_imsat.yaml \
    """


def _run_semi(*, save_dir: str, random_seed: int = 10, num_labeled_scan: int, max_epoch: int, num_batches: int,
              arch_checkpoint: str, lr: float, data_name: str = "acdc", imsat_weight: float, cons_weight: float,
              num_clusters: int, imsat_lamda: float, use_dynamic: str,
              num_subheads: int, head_type: str):
    return f""" python main_nd.py RandomSeed={random_seed} Trainer.name=semi \
     Trainer.save_dir={save_dir} Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} Data.name={data_name} \
    Data.labeled_scan_num={num_labeled_scan}  Arch.checkpoint={arch_checkpoint} Optim.lr={lr:.10f} \
    CrossCorrelationParameters.num_clusters={num_clusters}  \
    CrossCorrelationParameters.num_subheads={num_subheads}  \
    CrossCorrelationParameters.head_type={head_type}  \
    CrossCorrelationParameters.hooks.imsat.weight={imsat_weight:.10f} \
    CrossCorrelationParameters.hooks.imsat.lamda={imsat_lamda:.10f} \
    CrossCorrelationParameters.hooks.imsat.use_dynamic={use_dynamic} \
    CrossCorrelationParameters.hooks.consist.weight={cons_weight:.10f} \
    CrossCorrelationParameters.hooks.consist.weight={cons_weight:.10f} \
    --path   config/base.yaml  config/hooks/ccblocks_imsat.yaml   \
    """


def run_pretrain_ft(*, save_dir, random_seed: int = 10, max_epoch_pretrain: int, max_epoch: int, num_batches: int,
                    pretrain_scan_sample_num: int, data_name: str = "acdc", imsat_weight: float, cons_weight: float,
                    num_clusters: int, num_subheads: int, head_type: str, imsat_lamda: float, use_dynamic: str, ):
    data_opt = yaml_load(os.path.join(OPT_PATH, data_name + ".yaml"))
    labeled_scans = data_opt["labeled_ratios"][:-1]
    pretrain_save_dir = os.path.join(save_dir, "pretrain")
    pretrain_script = _run_pretrain_cc(
        save_dir=pretrain_save_dir, random_seed=random_seed, max_epoch=max_epoch_pretrain, num_batches=num_batches,
        lr=data_opt["pre_lr"], data_name=data_name, imsat_weight=imsat_weight, cons_weight=cons_weight,
        num_clusters=num_clusters, num_subheads=num_subheads, head_type=head_type, imsat_lamda=imsat_lamda,
        use_dynamic=use_dynamic, scan_sample_num=pretrain_scan_sample_num
    )
    ft_save_dir = os.path.join(save_dir, "tra")
    if data_name == "acdc" and not enable_acdc_all_class_train:
        run_ft = _run_ft_per_class
    else:
        run_ft = _run_ft
    ft_script = [
        run_ft(
            save_dir=os.path.join(ft_save_dir, f"labeled_num_{l:03d}"), random_seed=random_seed,
            num_labeled_scan=l, max_epoch=max_epoch, num_batches=num_batches,
            arch_checkpoint=f"{os.path.join(MODEL_PATH, pretrain_save_dir, 'last.pth')}",
            lr=data_opt["ft_lr"], data_name=data_name
        )
        for l in labeled_scans
    ]
    return [pretrain_script] + ft_script


def run_semi_regularize(
        *, save_dir, random_seed: int = 10, max_epoch: int, num_batches: int, data_name: str = "acdc",
        imsat_weight: float, cons_weight: float, num_clusters: int,
        num_subheads: int, head_type: str, imsat_lamda: float, use_dynamic: str,
) -> List[str]:
    data_opt = yaml_load(os.path.join(OPT_PATH, data_name + ".yaml"))
    labeled_scans = data_opt["labeled_ratios"][:-1]
    semi_script = [
        _run_semi(
            save_dir=os.path.join(save_dir, "semi", f"labeled_num_{l:03d}"), random_seed=random_seed,
            num_labeled_scan=l, max_epoch=max_epoch, num_batches=num_batches, arch_checkpoint="null",
            lr=data_opt["ft_lr"], data_name=data_name, imsat_weight=imsat_weight, cons_weight=cons_weight,
            num_clusters=num_clusters, num_subheads=num_subheads, head_type=head_type, imsat_lamda=imsat_lamda,
            use_dynamic=use_dynamic
        )
        for l in labeled_scans
    ]
    return semi_script


def run_pretrain_ft_with_grid_search(
        *, save_dir, random_seeds: Sequence[int] = 10, max_epoch_pretrain: int, max_epoch: int, num_batches: int,
        data_name: str, imsat_weight: Sequence[float], cons_weight: Sequence[float], num_clusters: Sequence[int],
        num_subheads: Sequence[int], head_type: Sequence[str], imsat_lamda: Sequence[float], use_dynamic: Sequence[str],
        max_num: Optional[int] = 200, pretrain_scan_sample_num: Sequence[int],
) -> Iterator[List[str]]:
    param_generator = grid_search(max_num=max_num, imsat_weight=imsat_weight, cons_weight=cons_weight,
                                  num_clusters=num_clusters, pretrain_scan_sample_num=pretrain_scan_sample_num,
                                  num_subheads=num_subheads, head_type=head_type,
                                  imsat_lamda=imsat_lamda,
                                  use_dynamic=use_dynamic,
                                  random_seed=random_seeds)
    for param in param_generator:
        random_seed = param.pop("random_seed")
        sp_str = get_hyper_param_string(**param)
        yield run_pretrain_ft(save_dir=os.path.join(save_dir, f"seed_{random_seed}", sp_str), random_seed=random_seed,
                              max_epoch=max_epoch, num_batches=num_batches, max_epoch_pretrain=max_epoch_pretrain,
                              data_name=data_name, **param)


def run_semi_regularize_with_grid_search(
        *, save_dir, random_seeds: Sequence[int] = 10, max_epoch: int, num_batches: int,
        data_name: str,
        imsat_weight: Sequence[float], cons_weight: Sequence[float], num_clusters: Sequence[int],
        num_subheads: Sequence[int], head_type: Sequence[str], imsat_lamda: Sequence[float], use_dynamic: Sequence[str],
        max_num: Optional[int] = 200,
) -> Iterator[List[str]]:
    param_generator = grid_search(max_num=max_num, imsat_weight=imsat_weight, cons_weight=cons_weight,
                                  num_clusters=num_clusters,
                                  num_subheads=num_subheads, head_type=head_type,
                                  imsat_lamda=imsat_lamda,
                                  use_dynamic=use_dynamic,
                                  random_seed=random_seeds)
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
    max_epoch_pretrain = args.max_epoch_pretrain
    num_batches = args.num_batches
    pretrain_scan_num = args.pretrain_scan_num
    save_dir = args.save_dir

    save_dir = os.path.join(save_dir, f"hash_{git_hash}/{data_name}/imsat")

    submitter = SlurmSubmitter(work_dir="../", stop_on_error=on_local, on_local=on_local)
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
        save_dir=os.path.join(save_dir, "pretrain"), random_seeds=random_seeds, max_epoch=max_epoch,
        num_batches=num_batches, data_name=data_name)
    jobs = list(job_generator)
    logger.info(f"logging {len(jobs)} jobs")
    for job in jobs:
        submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))

    # native imsat way with kl
    job_generator = run_pretrain_ft_with_grid_search(save_dir=os.path.join(save_dir, "pretrain"),
                                                     random_seeds=random_seeds, max_epoch=max_epoch,
                                                     num_batches=num_batches, max_epoch_pretrain=max_epoch_pretrain,
                                                     data_name=data_name, imsat_weight=(1,),
                                                     cons_weight=(0, 0.001, 0.01, 0.1, 1, 10),
                                                     num_clusters=[20, 40, 60],
                                                     max_num=500,
                                                     head_type=("linear",),
                                                     num_subheads=(3,),
                                                     imsat_lamda=(1,),
                                                     use_dynamic=("false",),
                                                     pretrain_scan_sample_num=(pretrain_scan_num,)
                                                     )
    jobs = list(job_generator)
    logger.info(f"logging {len(jobs)} jobs")
    for job in jobs:
        submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))

    # penalty based imsat way with kl
    job_generator = run_pretrain_ft_with_grid_search(save_dir=os.path.join(save_dir, "pretrain"),
                                                     random_seeds=random_seeds, max_epoch=max_epoch,
                                                     num_batches=num_batches, max_epoch_pretrain=max_epoch_pretrain,
                                                     data_name=data_name, imsat_weight=(1,),
                                                     cons_weight=(0, 0.01, 0.1, 1, 10),
                                                     num_clusters=(40),
                                                     max_num=500,
                                                     head_type=("linear",),
                                                     num_subheads=(3,),
                                                     imsat_lamda=(1,),
                                                     use_dynamic=("true",),
                                                     pretrain_scan_sample_num=(pretrain_scan_num,)
                                                     )
    jobs = list(job_generator)
    logger.info(f"logging {len(jobs)} jobs")
    for job in jobs:
        submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))
