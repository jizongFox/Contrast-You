import argparse
import os
from collections.abc import Iterable
from itertools import cycle
from typing import Sequence, List, Iterator, Optional

from loguru import logger

from contrastyou import __accounts, on_cc, MODEL_PATH, OPT_PATH, git_hash
from contrastyou.configure import yaml_load
from contrastyou.submitter import SlurmSubmitter
from script.utils import grid_search, move_dataset

enable_acdc_all_class_train = os.environ.get("ACDC_ALL_CLASS", "0") == "1"


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("save_dir", type=str, help="save dir")
    parser.add_argument(
        "--data-name", type=str,
        choices=("acdc", "acdc_lv", "acdc_rv", "acdc_myo", "prostate", "spleen", "hippocampus"),
        default="acdc",
        help="dataset_choice"
    )
    parser.add_argument("--max-epoch-pretrain", default=50, type=int, help="max epoch")
    parser.add_argument("--max-epoch", default=30, type=int, help="max epoch")
    parser.add_argument("--num-batches", default=300, type=int, help="number of batches")
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, ], )
    parser.add_argument("--pretrain-scan-num", type=int, default=6, help="default `scan_sample_num` for pretraining")
    parser.add_argument("--force-show", action="store_true", help="showing script")
    parser.add_argument("--pretrain", action="store_true", help="showing script")
    parser.add_argument("--semi", action="store_true", help="showing script")
    parser.add_argument("--baseline", action="store_true", help="showing script")

    args = parser.parse_args()
    return args


def get_hyper_param_string(**kwargs):
    def to_str(v):
        if isinstance(v, Iterable) and (not isinstance(v, str)):
            return "_".join([str(x) for x in v])
        return v

    list_string = [f"{k}_{to_str(v)}" for k, v in kwargs.items()]
    prefix = "/".join(list_string)
    return prefix


def _run_ft(*, save_dir: str, random_seed: int = 10, num_labeled_scan: int, max_epoch: int, num_batches: int,
            arch_checkpoint: str = "null", lr: float, data_name: str = "acdc"):
    return f""" python main.py RandomSeed={random_seed} Trainer.name=ft \
     Trainer.save_dir={save_dir} Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} Data.name={data_name} \
    Data.labeled_scan_num={num_labeled_scan}  Arch.checkpoint={arch_checkpoint} Optim.lr={lr:.10f} \
    """


def _run_ft_per_class(*, save_dir: str, random_seed: int = 10, num_labeled_scan: int, max_epoch: int, num_batches: int,
                      arch_checkpoint: str = "null", lr: float, data_name: str = "acdc"):
    assert data_name == "acdc", "only support acdc dataset"
    return f""" python main.py RandomSeed={random_seed} Trainer.name=ft \
     Trainer.save_dir={save_dir}/lv Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} Data.name={data_name}_lv \
    Data.labeled_scan_num={num_labeled_scan}  Arch.checkpoint={arch_checkpoint} Optim.lr={lr:.10f}  && \
    python main.py RandomSeed={random_seed} Trainer.name=ft \
     Trainer.save_dir={save_dir}/rv Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} Data.name={data_name}_rv \
    Data.labeled_scan_num={num_labeled_scan}  Arch.checkpoint={arch_checkpoint} Optim.lr={lr:.10f}  && \
    python main.py RandomSeed={random_seed} Trainer.name=ft \
     Trainer.save_dir={save_dir}/myo Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} Data.name={data_name}_myo \
    Data.labeled_scan_num={num_labeled_scan}  Arch.checkpoint={arch_checkpoint} Optim.lr={lr:.10f}  \
    """


def _run_semi(*, save_dir: str, random_seed: int = 10, num_labeled_scan: int, max_epoch: int, num_batches: int,
              arch_checkpoint: str, lr: float, data_name: str = "acdc", cc_weight: float,
              consistency_weight: float, power: float, head_type: str, num_subheads: int,
              num_clusters: int, kernel_size: int, rr_weight: float,
              rr_symmetric: str, rr_lamda: float, rr_alpha: float):
    return f""" python main_nd.py RandomSeed={random_seed} Trainer.name=semi \
     Trainer.save_dir={save_dir} Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} Data.name={data_name} \
    Data.labeled_scan_num={num_labeled_scan}  Arch.checkpoint={arch_checkpoint} Optim.lr={lr:.10f} \
    CrossCorrelationParameters.num_clusters={num_clusters}  \
    CrossCorrelationParameters.num_subheads={num_subheads}  \
    CrossCorrelationParameters.head_type={head_type}  \
    CrossCorrelationParameters.hooks.cc.weight={cc_weight:.10f}  \
    CrossCorrelationParameters.hooks.cc.kernel_size={kernel_size}  \
    CrossCorrelationParameters.hooks.cc.diff_power={power}  \
    CrossCorrelationParameters.hooks.rr.weight={rr_weight:.10f}  \
    CrossCorrelationParameters.hooks.rr.symmetric={rr_symmetric}  \
    CrossCorrelationParameters.hooks.rr.lamda={rr_lamda:.10f}  \
    CrossCorrelationParameters.hooks.rr.alpha={rr_alpha:.10f}  \
    ConsistencyParameters.weight={consistency_weight:.10f}  \
    --path   config/base.yaml  config/hooks/ccblocks2.yaml  config/hooks/consistency.yaml\
    """


def _run_multicore_semi(*, save_dir: str, random_seed: int = 10, num_labeled_scan: int, max_epoch: int,
                        num_batches: int,
                        arch_checkpoint: str, lr: float, data_name: str = "acdc", cc_weight: float,
                        consistency_weight: float, power: float, head_type: str,
                        num_subheads: int, mulitcore_multiplier: int, kernel_size: int, rr_weight: float,
                        rr_symmetric: str, rr_lamda: float, rr_alpha: float):
    return f""" python main_multicore.py RandomSeed={random_seed} Trainer.name=semi \
     Trainer.save_dir={save_dir} Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} Data.name={data_name} \
    Data.labeled_scan_num={num_labeled_scan}  Arch.checkpoint={arch_checkpoint} Optim.lr={lr:.10f} \
    CrossCorrelationParameters.feature_name=Deconv_1x1  \
    CrossCorrelationParameters.num_subheads={num_subheads}  \
    CrossCorrelationParameters.head_type={head_type}  \
    CrossCorrelationParameters.hooks.cc.weight={cc_weight:.10f}  \
    CrossCorrelationParameters.hooks.cc.diff_power={power}  \
    CrossCorrelationParameters.hooks.cc.kernel_size={kernel_size}  \
    CrossCorrelationParameters.hooks.rr.weight={rr_weight:.10f}  \
    CrossCorrelationParameters.hooks.rr.symmetric={rr_symmetric}  \
    CrossCorrelationParameters.hooks.rr.lamda={rr_lamda:.10f} \
    CrossCorrelationParameters.hooks.rr.alpha={rr_alpha:.10f}  \
    ConsistencyParameters.weight={consistency_weight:.10f}  \
    MulticoreParameters.multiplier={mulitcore_multiplier} \
    --path   config/base.yaml  config/hooks/ccblocks2.yaml config/hooks/multicore.yaml config/hooks/consistency.yaml\
    """


def _run_pretrain_cc(*, save_dir: str, random_seed: int = 10, max_epoch: int, num_batches: int, scan_sample_num: int,
                     cc_weight: float, consistency_weight: float, lr: float, data_name: str = "acdc", power: float,
                     head_type: str,
                     num_subheads: int, num_clusters: int,
                     kernel_size: int, rr_weight: float, rr_symmetric: str,
                     rr_lamda: float, rr_alpha: float):
    return f"""  python main_nd.py RandomSeed={random_seed} Trainer.name=pretrain_decoder Trainer.save_dir={save_dir} \
    Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches}  Optim.lr={lr:.10f} Data.name={data_name} \
    CrossCorrelationParameters.num_clusters={num_clusters}  \
    CrossCorrelationParameters.num_subheads={num_subheads}  \
    CrossCorrelationParameters.head_type={head_type}  \
    CrossCorrelationParameters.hooks.cc.weight={cc_weight:.10f}  \
    CrossCorrelationParameters.hooks.cc.kernel_size={kernel_size}  \
    CrossCorrelationParameters.hooks.cc.diff_power={power}  \
    ConsistencyParameters.weight={consistency_weight:.10f}  \
    CrossCorrelationParameters.hooks.rr.weight={rr_weight:.10f}  \
    CrossCorrelationParameters.hooks.rr.symmetric={rr_symmetric}  \
    CrossCorrelationParameters.hooks.rr.lamda={rr_lamda:.10f} \
    CrossCorrelationParameters.hooks.rr.alpha={rr_alpha:.10f}  \
    ContrastiveLoaderParams.scan_sample_num={scan_sample_num}  \
    --path config/base.yaml config/pretrain.yaml config/hooks/ccblocks2.yaml config/hooks/consistency.yaml\
    """


def run_pretrain_ft(*, save_dir, random_seed: int = 10, max_epoch_pretrain: int, max_epoch: int, num_batches: int,
                    data_name: str = "acdc", pretrain_scan_sample_num: int,
                    cc_weight, consistency_weight,
                    power: float, head_type: str, num_subheads: int, num_clusters: int, kernel_size: int,
                    rr_weight: float, rr_symmetric: str, rr_lamda: float,
                    rr_alpha: float
                    ):
    data_opt = yaml_load(os.path.join(OPT_PATH, data_name + ".yaml"))
    labeled_scans = data_opt["labeled_ratios"][:-1]
    pretrain_save_dir = os.path.join(save_dir, "pretrain")
    pretrain_script = _run_pretrain_cc(
        save_dir=pretrain_save_dir, random_seed=random_seed, max_epoch=max_epoch_pretrain, num_batches=num_batches,
        cc_weight=cc_weight, lr=data_opt["pre_lr"], data_name=data_name,
        consistency_weight=consistency_weight, power=power, head_type=head_type,
        num_subheads=num_subheads, num_clusters=num_clusters, kernel_size=kernel_size,
        rr_weight=rr_weight, rr_symmetric=rr_symmetric, rr_lamda=rr_lamda, rr_alpha=rr_alpha,
        scan_sample_num=pretrain_scan_sample_num
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
        cc_weight: float, consistency_weight: float, power: float, head_type: str,
        num_subheads: int, num_clusters: int, kernel_size: int, rr_weight: float,
        rr_symmetric: str, rr_lamda: float, rr_alpha: float
) -> List[str]:
    data_opt = yaml_load(os.path.join(OPT_PATH, data_name + ".yaml"))
    labeled_scans = data_opt["labeled_ratios"][:-1]
    semi_script = [
        _run_semi(
            save_dir=os.path.join(save_dir, "semi", f"labeled_num_{l:03d}"), random_seed=random_seed,
            num_labeled_scan=l, max_epoch=max_epoch, num_batches=num_batches, arch_checkpoint="null",
            lr=data_opt["ft_lr"], data_name=data_name,
            cc_weight=cc_weight, consistency_weight=consistency_weight, power=power,
            head_type=head_type, num_subheads=num_subheads, num_clusters=num_clusters, kernel_size=kernel_size,
            rr_weight=rr_weight, rr_symmetric=rr_symmetric,
            rr_lamda=rr_lamda, rr_alpha=rr_alpha
        )
        for l in labeled_scans
    ]
    return semi_script


def run_baseline(
        *, save_dir, random_seed: int = 10, max_epoch: int, num_batches: int, data_name: str = "acdc",
        arch_checkpoint: str = "null"
) -> List[str]:
    data_opt = yaml_load(os.path.join(OPT_PATH, data_name + ".yaml"))
    labeled_scans = data_opt["labeled_ratios"][:-1]
    if data_name == "acdc" and not enable_acdc_all_class_train:
        run_ft = _run_ft_per_class
    else:
        run_ft = _run_ft
    ft_script = [
        run_ft(
            save_dir=os.path.join(save_dir, "baseline", f"labeled_num_{l:03d}"), random_seed=random_seed,
            num_labeled_scan=l, max_epoch=max_epoch, num_batches=num_batches,
            lr=data_opt["ft_lr"], data_name=data_name, arch_checkpoint=arch_checkpoint,
        )
        for l in labeled_scans
    ]
    return ft_script


def run_baseline_with_grid_search(*, save_dir, random_seeds: Sequence[int] = 10, max_epoch: int, num_batches: int,
                                  data_name: str = "acdc", arch_checkpoint: str = "null"):
    rand_seed_gen = grid_search(random_seed=random_seeds)
    for random_seed in rand_seed_gen:
        yield run_baseline(save_dir=os.path.join(save_dir, f"seed_{random_seed['random_seed']}"),
                           **random_seed, max_epoch=max_epoch, num_batches=num_batches,
                           data_name=data_name, arch_checkpoint=arch_checkpoint)


def run_pretrain_ft_with_grid_search(
        *, save_dir, random_seeds: Sequence[int] = 10, max_epoch_pretrain: int, max_epoch: int, num_batches: int,
        data_name: str, cc_weights: Sequence[float], consistency_weights: Sequence[float],
        powers: Sequence[float], head_types=Sequence[str],
        num_subheads: Sequence[int], num_clusters: Sequence[int], kernel_size: Sequence[int],
        rr_weight: Sequence[float],
        rr_symmetric: Sequence[str], rr_lamda: Sequence[float], rr_alpha: Sequence[float],
        max_num: Optional[int] = 200, pretrain_scan_sample_num: Sequence[int],
) -> Iterator[List[str]]:
    param_generator = grid_search(max_num=max_num, cc_weight=cc_weights,
                                  random_seed=random_seeds,
                                  pretrain_scan_sample_num=pretrain_scan_sample_num,
                                  consistency_weight=consistency_weights, rr_weight=rr_weight,
                                  power=powers, head_type=head_types, num_subheads=num_subheads,
                                  kernel_size=kernel_size, rr_symmetric=rr_symmetric,
                                  num_clusters=num_clusters, rr_lamda=rr_lamda, rr_alpha=rr_alpha)
    for param in param_generator:
        random_seed = param.pop("random_seed")
        sp_str = get_hyper_param_string(**param)
        yield run_pretrain_ft(save_dir=os.path.join(save_dir, f"seed_{random_seed}", sp_str), random_seed=random_seed,
                              max_epoch=max_epoch, num_batches=num_batches, max_epoch_pretrain=max_epoch_pretrain,
                              data_name=data_name, **param)


def run_semi_regularize_with_grid_search(
        *, save_dir, random_seeds: Sequence[int] = 10, max_epoch: int, num_batches: int,
        data_name: str,
        cc_weights: Sequence[float], consistency_weights: Sequence[float],
        powers: Sequence[float], head_types: Sequence[str],
        num_subheads: Sequence[int], num_clusters: Sequence[int], kernel_size: Sequence[int],
        rr_weight: Sequence[float],
        rr_symmetric: Sequence[str], rr_lamda: Sequence[float], rr_alpha: Sequence[float],
        max_num: Optional[int] = 200,
) -> Iterator[List[str]]:
    param_generator = grid_search(cc_weight=cc_weights,
                                  random_seed=random_seeds, consistency_weight=consistency_weights, rr_weight=rr_weight,
                                  power=powers, head_type=head_types,
                                  num_subheads=num_subheads, num_clusters=num_clusters, max_num=max_num,
                                  kernel_size=kernel_size, rr_symmetric=rr_symmetric,
                                  rr_lamda=rr_lamda, rr_alpha=rr_alpha
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
    max_epoch_pretrain = args.max_epoch_pretrain
    num_batches = args.num_batches
    pretrain_scan_num = args.pretrain_scan_num
    save_dir = args.save_dir

    if "acdc" in data_name:
        power = (0.75,)
    elif "prostate" in data_name:
        power = (0.5,)
    else:
        raise NotImplementedError("powers")

    save_dir = os.path.join(save_dir, f"hash_{git_hash}/{data_name}")

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

    if args.baseline:
        # baseline
        job_generator = run_baseline_with_grid_search(
            save_dir=os.path.join(save_dir, "pretrain"), random_seeds=random_seeds, max_epoch=max_epoch,
            num_batches=num_batches, data_name=data_name)

        jobs = list(job_generator)
        logger.info(f"logging {len(jobs)} jobs")
        for job in jobs:
            submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))

    if args.pretrain:
        # use only rr
        job_generator = run_pretrain_ft_with_grid_search(save_dir=os.path.join(save_dir, "pretrain"),
                                                         random_seeds=random_seeds, max_epoch=max_epoch,
                                                         num_batches=num_batches, max_epoch_pretrain=max_epoch_pretrain,
                                                         data_name=data_name,
                                                         cc_weights=[0, 0.1, 0.5, 1, 2],
                                                         consistency_weights=[0],
                                                         powers=power,
                                                         head_types="linear",
                                                         num_subheads=(3,),
                                                         num_clusters=[20, 40, 60],
                                                         max_num=500,
                                                         kernel_size=5,
                                                         rr_weight=(1,),
                                                         rr_symmetric="true",
                                                         rr_lamda=(1,),
                                                         rr_alpha=(0, 0.25, 0.5, 0.75, 1),
                                                         pretrain_scan_sample_num=(pretrain_scan_num,)
                                                         )
        jobs = list(job_generator)
        logger.info(f"logging {len(jobs)} jobs")
        for job in jobs:
            submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))

    if args.semi:
        # only with RR on semi supervised case
        job_generator = run_semi_regularize_with_grid_search(save_dir=os.path.join(save_dir, "semi"),
                                                             random_seeds=random_seeds, max_epoch=max_epoch,
                                                             num_batches=num_batches,
                                                             data_name=data_name,
                                                             cc_weights=[0, 0.1, 0.5, 1],
                                                             consistency_weights=[0, 0.1, 1],
                                                             powers=power,
                                                             head_types="linear",
                                                             num_subheads=(3,),
                                                             num_clusters=[20, 40, 60],
                                                             max_num=500,
                                                             kernel_size=5,
                                                             rr_weight=(1,),
                                                             rr_symmetric="true",
                                                             rr_lamda=(1,),
                                                             rr_alpha=(0.5,),
                                                             )

        jobs = list(job_generator)
        logger.info(f"logging {len(jobs)} jobs")
        for job in jobs:
            submitter.submit(" && \n ".join(job), force_show=force_show, time=8, account=next(account))
