import argparse
import os
from collections.abc import Iterable
from itertools import cycle
from typing import Sequence, List, Iterator, Optional

from loguru import logger

from contrastyou import __accounts, on_cc, MODEL_PATH, OPT_PATH, git_hash
from contrastyou.configure import yaml_load
from contrastyou.submitter import SlurmSubmitter
from script.script_generator_pretrain_cc import _run_ft, _run_ft_per_class
from script.utils import grid_search, move_dataset


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("save_dir", type=str, help="save dir")
    parser.add_argument("--data-name", type=str, choices=("acdc", "acdc_lv", "acdc_rv", "prostate"), default="acdc",
                        help="dataset_choice")
    parser.add_argument("--max-epoch-pretrain", default=50, type=int, help="max epoch")
    parser.add_argument("--max-epoch", default=30, type=int, help="max epoch")
    parser.add_argument("--num-batches", default=300, type=int, help="number of batches")
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, ], )
    parser.add_argument("--force-show", action="store_true", help="showing script")
    parser.add_argument("--encoder", action="store_true", default=False, help="enable encoder pretraining")
    parser.add_argument("--decoder", action="store_true", default=False, help="enable decoder pretraining")

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


def _run_pretrain_cc(*, save_dir: str, random_seed: int = 10, max_epoch: int, num_batches: int, lr: float,
                     data_name: str = "acdc", infonce_encoder_weight: float, infonce_decoder_weight: float,
                     decoder_spatial_size: int):
    return f"""  python main_nd.py RandomSeed={random_seed} Trainer.name=pretrain_decoder Trainer.save_dir={save_dir} \
    Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches}  Optim.lr={lr:.10f} Data.name={data_name} \
    InfonceParams.weights=[{infonce_encoder_weight:.10f},{infonce_decoder_weight:.10f}] \
    InfonceParams.spatial_size=[1,{decoder_spatial_size}] \
    --path config/base.yaml config/pretrain.yaml config/hooks/infonce_encoder_dense.yaml \
    """


def _run_semi(*, save_dir: str, random_seed: int = 10, num_labeled_scan: int, max_epoch: int, num_batches: int,
              arch_checkpoint: str, lr: float, data_name: str = "acdc", infonce_encoder_weight: float,
              infonce_decoder_weight: float, decoder_spatial_size: int):
    return f""" python main_nd.py RandomSeed={random_seed} Trainer.name=semi \
     Trainer.save_dir={save_dir} Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} Data.name={data_name} \
    Data.labeled_scan_num={num_labeled_scan}  Arch.checkpoint={arch_checkpoint} Optim.lr={lr:.10f} \
    InfonceParams.weights=[{infonce_encoder_weight:.10f},{infonce_decoder_weight:.10f}] \
    InfonceParams.spatial_size=[1,{decoder_spatial_size}] \
    --path   config/base.yaml  config/hooks/infonce_encoder_dense.yaml  \
    """


def run_pretrain_ft(*, save_dir, random_seed: int = 10, max_epoch_pretrain: int, max_epoch: int, num_batches: int,
                    data_name: str = "acdc", infonce_encoder_weight: float, infonce_decoder_weight: float,
                    decoder_spatial_size: int
                    ):
    data_opt = yaml_load(os.path.join(OPT_PATH, data_name + ".yaml"))
    labeled_scans = data_opt["labeled_ratios"][:-1]
    pretrain_save_dir = os.path.join(save_dir, "pretrain")
    pretrain_script = _run_pretrain_cc(
        save_dir=pretrain_save_dir, random_seed=random_seed, max_epoch=max_epoch_pretrain, num_batches=num_batches,
        lr=data_opt["pre_lr"], data_name=data_name, infonce_decoder_weight=infonce_decoder_weight,
        infonce_encoder_weight=infonce_encoder_weight,
        decoder_spatial_size=decoder_spatial_size
    )
    ft_save_dir = os.path.join(save_dir, "tra")
    if data_name == "acdc":
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
        infonce_encoder_weight: float, infonce_decoder_weight: float, decoder_spatial_size: int
) -> List[str]:
    data_opt = yaml_load(os.path.join(OPT_PATH, data_name + ".yaml"))
    labeled_scans = data_opt["labeled_ratios"][:-1]
    semi_script = [
        _run_semi(
            save_dir=os.path.join(save_dir, "semi", f"labeled_num_{l:03d}"), random_seed=random_seed,
            num_labeled_scan=l, max_epoch=max_epoch, num_batches=num_batches, arch_checkpoint="null",
            lr=data_opt["ft_lr"], data_name=data_name, infonce_encoder_weight=infonce_encoder_weight,
            infonce_decoder_weight=infonce_decoder_weight,
            decoder_spatial_size=decoder_spatial_size
        )
        for l in labeled_scans
    ]
    return semi_script


def run_baseline(
        *, save_dir, random_seed: int = 10, max_epoch: int, num_batches: int, data_name: str = "acdc"
) -> List[str]:
    data_opt = yaml_load(os.path.join(OPT_PATH, data_name + ".yaml"))
    labeled_scans = data_opt["labeled_ratios"][:-1]
    if data_name == "acdc":
        run_ft = _run_ft_per_class
    else:
        run_ft = _run_ft
    ft_script = [
        run_ft(
            save_dir=os.path.join(save_dir, "baseline", f"labeled_num_{l:03d}"), random_seed=random_seed,
            num_labeled_scan=l, max_epoch=max_epoch, num_batches=num_batches,
            arch_checkpoint="null",
            lr=data_opt["ft_lr"], data_name=data_name
        )
        for l in labeled_scans
    ]
    return ft_script


def run_pretrain_ft_with_grid_search(
        *, save_dir, random_seeds: Sequence[int] = 10, max_epoch_pretrain: int, max_epoch: int, num_batches: int,
        data_name: str, infonce_decoder_weight: Sequence[float], infonce_encoder_weight: Sequence[float],
        decoder_spatial_size: Sequence[int],
        include_baseline=True, max_num: Optional[int] = 200,
) -> Iterator[List[str]]:
    param_generator = grid_search(max_num=max_num, infonce_decoder_weight=infonce_decoder_weight,
                                  infonce_encoder_weight=infonce_encoder_weight,
                                  decoder_spatial_size=decoder_spatial_size,
                                  random_seed=random_seeds)
    for param in param_generator:
        random_seed = param.pop("random_seed")
        sp_str = get_hyper_param_string(**param)
        yield run_pretrain_ft(save_dir=os.path.join(save_dir, f"seed_{random_seed}", sp_str), random_seed=random_seed,
                              max_epoch=max_epoch, num_batches=num_batches, max_epoch_pretrain=max_epoch_pretrain,
                              data_name=data_name, **param)

    if include_baseline:
        rand_seed_gen = grid_search(random_seed=random_seeds)
        for random_seed in rand_seed_gen:
            yield run_baseline(save_dir=os.path.join(save_dir, f"seed_{random_seed['random_seed']}"),
                               **random_seed, max_epoch=max_epoch, num_batches=num_batches,
                               data_name=data_name)


def run_semi_regularize_with_grid_search(
        *, save_dir, random_seeds: Sequence[int] = 10, max_epoch: int, num_batches: int,
        data_name: str, infonce_encoder_weight: Sequence[float],
        infonce_decoder_weight: Sequence[float], decoder_spatial_size: Sequence[int],
        include_baseline=True, max_num: Optional[int] = 200,
) -> Iterator[List[str]]:
    param_generator = grid_search(
        max_num=max_num, decoder_spatial_size=decoder_spatial_size,
        infonce_decoder_weight=infonce_decoder_weight,
        infonce_encoder_weight=infonce_encoder_weight,
        random_seed=random_seeds
    )
    for param in param_generator:
        random_seed = param.pop("random_seed")
        sp_str = get_hyper_param_string(**param)
        yield run_semi_regularize(save_dir=os.path.join(save_dir, f"seed_{random_seed}", sp_str),
                                  random_seed=random_seed,
                                  max_epoch=max_epoch, num_batches=num_batches, data_name=data_name, **param)

    if include_baseline:
        rand_seed_gen = grid_search(random_seed=random_seeds)
        for random_seed in rand_seed_gen:
            yield run_baseline(save_dir=os.path.join(save_dir, f"seed_{random_seed['random_seed']}"),
                               **random_seed, max_epoch=max_epoch, num_batches=num_batches,
                               data_name=data_name)


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

    save_dir = args.save_dir

    save_dir = os.path.join(save_dir, f"hash_{git_hash}/{data_name}/infonce")

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

    if args.encoder:
        # only with encoder
        job_generator = run_pretrain_ft_with_grid_search(
            save_dir=os.path.join(save_dir, "pretrain", "encoder"),
            random_seeds=random_seeds, max_epoch=max_epoch,
            num_batches=num_batches, max_epoch_pretrain=max_epoch_pretrain,
            data_name=data_name, infonce_decoder_weight=(0,),
            infonce_encoder_weight=(1,),
            decoder_spatial_size=(10,),
            include_baseline=True, max_num=500
        )
        jobs = list(job_generator)
        logger.info(f"logging {len(jobs)} jobs")
        for job in jobs:
            submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))
    if args.decoder:
        # only with decoder
        job_generator = run_pretrain_ft_with_grid_search(
            save_dir=os.path.join(save_dir, "pretrain", "decoder"),
            random_seeds=random_seeds, max_epoch=max_epoch,
            num_batches=num_batches, max_epoch_pretrain=max_epoch_pretrain,
            data_name=data_name, infonce_decoder_weight=(1,),
            infonce_encoder_weight=(0,),
            decoder_spatial_size=(20,),
            include_baseline=True, max_num=500
        )
        jobs = list(job_generator)
        logger.info(f"logging {len(jobs)} jobs")
        for job in jobs:
            submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))

    if args.encoder and args.decoder:
        # encoder + decoder
        job_generator = run_pretrain_ft_with_grid_search(
            save_dir=os.path.join(save_dir, "pretrain", "encoder_decoder"),
            random_seeds=random_seeds, max_epoch=max_epoch,
            num_batches=num_batches, max_epoch_pretrain=max_epoch_pretrain,
            data_name=data_name, infonce_decoder_weight=(0.0001, 0.001, 0.01, 0.1, 1, 10),
            infonce_encoder_weight=(0.1, 1,),
            decoder_spatial_size=(20,),
            include_baseline=True, max_num=500
        )
        jobs = list(job_generator)
        logger.info(f"logging {len(jobs)} jobs")
        for job in jobs:
            submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))

    # semi
    if 0:
        # only with encoder
        job_generator = run_semi_regularize_with_grid_search(
            save_dir=os.path.join(save_dir, "semi", "encoder"),
            random_seeds=random_seeds, max_epoch=max_epoch,
            num_batches=num_batches,
            data_name=data_name, infonce_decoder_weight=(0,),
            infonce_encoder_weight=(0.0001, 0.001, 0.01),
            decoder_spatial_size=(10,),
            include_baseline=True, max_num=500
        )
        jobs = list(job_generator)
        logger.info(f"logging {len(jobs)} jobs")
        for job in jobs:
            submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))

        # only with decoder
        job_generator = run_semi_regularize_with_grid_search(
            save_dir=os.path.join(save_dir, "semi", "decoder"),
            random_seeds=random_seeds, max_epoch=max_epoch,
            num_batches=num_batches,
            data_name=data_name, infonce_decoder_weight=(0.0001, 0.001, 0.01),
            infonce_encoder_weight=(0,),
            decoder_spatial_size=(20,),
            include_baseline=True, max_num=500
        )
        jobs = list(job_generator)
        logger.info(f"logging {len(jobs)} jobs")
        for job in jobs:
            submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))

        # encoder + decoder
        job_generator = run_semi_regularize_with_grid_search(
            save_dir=os.path.join(save_dir, "semi", "encoder_decoder"),
            random_seeds=random_seeds, max_epoch=max_epoch,
            num_batches=num_batches,
            data_name=data_name, infonce_decoder_weight=(0.0001, 0.001, 0.01),
            infonce_encoder_weight=(0.0001, 0.001, 0.01),
            decoder_spatial_size=(20,),
            include_baseline=True, max_num=500
        )
        jobs = list(job_generator)
        logger.info(f"logging {len(jobs)} jobs")
        for job in jobs:
            submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))