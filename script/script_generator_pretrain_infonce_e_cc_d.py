import argparse
import os
from itertools import cycle
from typing import Sequence, List, Iterator, Optional

from loguru import logger

from contrastyou import __accounts, on_cc, MODEL_PATH, OPT_PATH, git_hash
from contrastyou.configure import yaml_load
from contrastyou.submitter2 import SlurmSubmitter
from contrastyou.utils import deprecated
from script import utils
from script.script_generator_pretrain_cc import get_hyper_param_string, _run_ft_per_class, _run_ft, \
    run_baseline_with_grid_search
from script.utils import grid_search, move_dataset


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("save_dir", type=str, help="save dir")
    parser.add_argument("--data-name", type=str,
                        choices=(
                            "acdc", "acdc_lv", "acdc_rv", "acdc_myo", "prostate", "spleen", "hippocampus", "mmwhsct",
                            "mmwhsmr"),
                        default="acdc",
                        help="dataset_choice")
    parser.add_argument("--enable_acdc_all_class_train", action="store_true", help="enable acdc all class train",
                        default=False)
    parser.add_argument("--max-epoch-pretrain", default=50, type=int, help="max epoch")
    parser.add_argument("--max-epoch", default=30, type=int, help="max epoch")
    parser.add_argument("--num-batches", default=300, type=int, help="number of batches")
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, ], )
    parser.add_argument("--pretrain-scan-num", type=int, default=6, help="default `scan_sample_num` for pretraining")
    parser.add_argument("--force-show", action="store_true", help="showing script")
    parser.add_argument("--pretrain", action="store_true", help="showing script")
    parser.add_argument("--baseline", action="store_true", help="showing script")
    args = parser.parse_args()
    return args


@deprecated
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
    --path   config/base.yaml  config/hooks/ccblocks2.yaml  config/hooks/consistency.yaml config/hooks/infonce_encoder.yaml \
    """


def _run_pretrain_cc(*, save_dir: str, random_seed: int = 10, max_epoch: int, num_batches: int, scan_sample_num: int,
                     cc_weight: float, consistency_weight: float, lr: float, data_name: str = "acdc", power: float,
                     head_type: str,
                     num_subheads: int, num_clusters: int,
                     kernel_size: int, rr_weight: float, rr_symmetric: str,
                     rr_lamda: float, rr_alpha: float):
    return f"""  python main_nd.py -o RandomSeed={random_seed} Trainer.name=pretrain_decoder Trainer.save_dir={save_dir} \
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
    --path config/base.yaml config/pretrain.yaml config/hooks/ccblocks2.yaml config/hooks/consistency.yaml config/hooks/infonce_encoder.yaml \
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
    if data_name == "acdc" and not utils.enable_acdc_all_class_train:
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


@deprecated
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
    utils.enable_acdc_all_class_train = args.enable_acdc_all_class_train
    account = cycle(__accounts)
    on_local = not on_cc()
    force_show = args.force_show
    data_name = args.data_name
    random_seeds = args.seeds
    max_epoch = args.max_epoch
    max_epoch_pretrain = args.max_epoch_pretrain
    num_batches = args.num_batches
    pretrain_scan_num = args.pretrain_scan_num

    if "acdc" in data_name:  # for all subclasses of `acdc` dataset
        power = (0.75,)
    else:
        power = [1, 1.25, 1.5]

    save_dir = args.save_dir

    save_dir = os.path.join(save_dir, f"hash_{git_hash}/{data_name}")

    submitter = SlurmSubmitter(stop_on_error=True, verbose=True, dry_run=force_show, on_local=not on_cc())
    submitter.set_startpoint_path("../")
    submitter.set_prepare_scripts(
        *[
            "module load python/3.8.2 ", "source ~/venv/bin/activate ",
            move_dataset(), "nvidia-smi",
            "python -c 'import torch; print(torch.randn(1,1,1,1,device=\"cuda\"))'"
        ])

    submitter.update_env_params(
        PYTHONWARNINGS="ignore",
        CUBLAS_WORKSPACE_CONFIG=":16:8",
        LOGURU_LEVEL="TRACE",
        OMP_NUM_THREADS=1,
        PYTHONOPTIMIZE=1
    )

    submitter.update_sbatch_params(mem=24)
    # baseline
    if args.baseline:
        job_generator = run_baseline_with_grid_search(
            save_dir=os.path.join(save_dir, "baseline"), random_seeds=random_seeds, max_epoch=max_epoch,
            num_batches=num_batches, data_name=data_name)
        jobs = list(job_generator)
        logger.info(f"logging {len(jobs)} jobs")
        for job in jobs:
            submitter.submit(" && \n ".join(job), time=4, )

    # use only rr
    if args.pretrain:
        job_generator = run_pretrain_ft_with_grid_search(save_dir=os.path.join(save_dir, "pretrain"),
                                                         random_seeds=random_seeds, max_epoch=max_epoch,
                                                         num_batches=num_batches, max_epoch_pretrain=max_epoch_pretrain,
                                                         data_name=data_name,
                                                         cc_weights=[1],
                                                         consistency_weights=[0],
                                                         powers=power,
                                                         head_types="linear",
                                                         num_subheads=(3,),
                                                         num_clusters=[40],
                                                         max_num=500,
                                                         kernel_size=5,
                                                         rr_weight=(1,),
                                                         rr_symmetric="true",
                                                         rr_lamda=(1,),
                                                         rr_alpha=(0.5,),
                                                         pretrain_scan_sample_num=(pretrain_scan_num,)
                                                         )
        jobs = list(job_generator)
        logger.info(f"logging {len(jobs)} jobs")
        for job in jobs:
            submitter.submit(" && \n ".join(job), time=4, )
