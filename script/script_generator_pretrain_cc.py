import argparse
import os
from collections.abc import Iterable
from itertools import cycle
from typing import Sequence, List, Iterator

from contrastyou import __accounts, on_cc, MODEL_PATH, OPT_PATH, git_hash
from contrastyou.configure import yaml_load
from contrastyou.submitter import SlurmSubmitter
from script.utils import grid_search, move_dataset

parser = argparse.ArgumentParser()
parser.add_argument("save_dir", type=str, help="save dir")
args = parser.parse_args()

account = cycle(__accounts)
on_local = not on_cc()
force_show = True
data_name = "acdc"
save_dir = args.save_dir

save_dir = os.path.join(save_dir, f"hash_{git_hash}/{data_name}")


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


def _run_semi(*, save_dir: str, random_seed: int = 10, num_labeled_scan: int, max_epoch: int, num_batches: int,
              arch_checkpoint: str, lr: float, data_name: str = "acdc", cc_weight: float, mi_weight: float,
              consistency_weight: float, padding: int, lamda: float, power: float):
    return f""" python main_nd.py RandomSeed={random_seed} Trainer.name=semi \
     Trainer.save_dir={save_dir} Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} Data.name={data_name} \
    Data.labeled_scan_num={num_labeled_scan}  Arch.checkpoint={arch_checkpoint} Optim.lr={lr:.10f} \
    CrossCorrelationParameters.mi_weights={mi_weight:.10f}  \
    CrossCorrelationParameters.cc_weights={cc_weight:.10f}  \
    ConsistencyParameters.weight={consistency_weight:.10f}  \
    CrossCorrelationParameters.IID.padding={padding}  \
    CrossCorrelationParameters.IID.lamda={lamda:.10f} \
    CrossCorrelationParameters.norm.power={power:.10f}  \
    --path   config/base.yaml  config/hooks/ccblocks.yaml  config/hooks/consistency.yaml\
    """


def _run_pretrain_cc(*, save_dir: str, random_seed: int = 10, max_epoch: int, num_batches: int, cc_weight: float,
                     mi_weight: float, consistency_weight: float, lr: float, data_name: str = "acdc", padding: int,
                     lamda: float, power: float):
    return f"""  python main_nd.py RandomSeed={random_seed} Trainer.name=pretrain_decoder Trainer.save_dir={save_dir} \
    Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches} CrossCorrelationParameters.mi_weights={mi_weight:.10f}  \
    CrossCorrelationParameters.cc_weights={cc_weight:.10f}  Optim.lr={lr:.10f} Data.name={data_name} \
    ConsistencyParameters.weight={consistency_weight:.10f}  \
    CrossCorrelationParameters.IID.padding={padding}  \
    CrossCorrelationParameters.IID.lamda={lamda:.10f} \
    CrossCorrelationParameters.norm.power={power:.10f}  \
    --path config/base.yaml config/pretrain.yaml config/hooks/ccblocks.yaml config/hooks/consistency.yaml\
    """


def run_pretrain_ft(*, save_dir, random_seed: int = 10, max_epoch: int, num_batches: int, data_name: str = "acdc",
                    mi_weight, cc_weight, consistency_weight, padding: int,
                    lamda: float, power: float):
    data_opt = yaml_load(os.path.join(OPT_PATH, data_name + ".yaml"))
    labeled_scans = data_opt["labeled_ratios"][:-1]
    pretrain_save_dir = os.path.join(save_dir, "pretrain")
    pretrain_script = _run_pretrain_cc(
        save_dir=pretrain_save_dir, random_seed=random_seed, max_epoch=max_epoch, num_batches=num_batches,
        mi_weight=mi_weight, cc_weight=cc_weight, lr=data_opt["pre_lr"], data_name=data_name,
        consistency_weight=consistency_weight, padding=padding, lamda=lamda, power=power
    )
    ft_save_dir = os.path.join(save_dir, "tra")
    ft_script = [
        _run_ft(
            save_dir=os.path.join(ft_save_dir, f"labeled_num_{l:03d}"), random_seed=random_seed,
            num_labeled_scan=l, max_epoch=max_epoch, num_batches=num_batches,
            arch_checkpoint=f"{os.path.join(MODEL_PATH, pretrain_save_dir, 'last.pth')}",
            lr=data_opt["ft_lr"], data_name=data_name
        )
        for l in labeled_scans
    ]
    return [pretrain_script] + ft_script


def run_semi_regularize(*, save_dir, random_seed: int = 10, max_epoch: int, num_batches: int, data_name: str = "acdc",
                        mi_weight: float, cc_weight: float, consistency_weight: float, padding: int,
                        lamda: float, power: float) -> List[str]:
    data_opt = yaml_load(os.path.join(OPT_PATH, data_name + ".yaml"))
    labeled_scans = data_opt["labeled_ratios"][:-1]
    semi_script = [
        _run_semi(
            save_dir=os.path.join(save_dir, "semi", f"labeled_num_{l:03d}"), random_seed=random_seed,
            num_labeled_scan=l, max_epoch=max_epoch, num_batches=num_batches, arch_checkpoint="null",
            lr=data_opt["ft_lr"], data_name=data_name, mi_weight=mi_weight,
            cc_weight=cc_weight, consistency_weight=consistency_weight, padding=padding, lamda=lamda, power=power)
        for l in labeled_scans
    ]
    return semi_script


def run_baseline(
    *, save_dir, random_seed: int = 10, max_epoch: int, num_batches: int, data_name: str = "acdc"
) -> List[str]:
    data_opt = yaml_load(os.path.join(OPT_PATH, data_name + ".yaml"))
    labeled_scans = data_opt["labeled_ratios"][:-1]
    ft_script = [
        _run_ft(
            save_dir=os.path.join(save_dir, "baseline", f"labeled_num_{l:03d}"), random_seed=random_seed,
            num_labeled_scan=l, max_epoch=max_epoch, num_batches=num_batches,
            arch_checkpoint="null",
            lr=data_opt["ft_lr"], data_name=data_name
        )
        for l in labeled_scans
    ]
    return ft_script


def run_pretrain_ft_with_grid_search(
    *, save_dir, random_seeds: Sequence[int] = 10, max_epoch: int, num_batches: int,
    data_name: str,
    mi_weights: Sequence[float], cc_weights: Sequence[float], consistency_weights: Sequence[float],
    paddings: Sequence[int], lamdas: Sequence[float], powers: Sequence[float], include_baseline=True
) -> Iterator[List[str]]:
    param_generator = grid_search(mi_weight=mi_weights, cc_weight=cc_weights, random_seed=random_seeds,
                                  consistency_weight=consistency_weights, padding=paddings, lamda=lamdas,
                                  power=powers)
    for param in param_generator:
        random_seed = param.pop("random_seed")
        sp_str = get_hyper_param_string(**param)
        yield run_pretrain_ft(save_dir=os.path.join(save_dir, f"seed_{random_seed}", sp_str), random_seed=random_seed,
                              max_epoch=max_epoch, num_batches=num_batches, data_name=data_name, **param)

    if include_baseline:
        rand_seed_gen = grid_search(random_seed=random_seeds)
        for random_seed in rand_seed_gen:
            yield run_baseline(save_dir=os.path.join(save_dir, f"seed_{random_seed['random_seed']}"),
                               **random_seed, max_epoch=max_epoch, num_batches=num_batches,
                               data_name=data_name)


def run_semi_regularize_with_grid_search(
    *, save_dir, random_seeds: Sequence[int] = 10, max_epoch: int, num_batches: int,
    data_name: str,
    mi_weights: Sequence[float], cc_weights: Sequence[float], consistency_weights: Sequence[float],
    paddings: Sequence[int], lamdas: Sequence[float], powers: Sequence[float],
    include_baseline=True
) -> Iterator[List[str]]:
    param_generator = grid_search(mi_weight=mi_weights, cc_weight=cc_weights, random_seed=random_seeds,
                                  consistency_weight=consistency_weights, padding=paddings, lamda=lamdas,
                                  power=powers)
    for param in param_generator:
        random_seed = param.pop("random_seed")
        sp_str = get_hyper_param_string(**param)
        yield run_semi_regularize(save_dir=os.path.join(save_dir, f"seed_{random_seed}", sp_str),
                                  random_seed=random_seed,
                                  max_epoch=max_epoch, num_batches=num_batches, data_name=data_name, **param)

    if include_baseline:
        rand_seed_gen = grid_search(random_seed=random_seeds)
        for random_seed in rand_seed_gen:
            yield run_baseline(save_dir=os.path.join(save_dir, f"seed_{random_seed['random_seed']}", "baseline"),
                               **random_seed, max_epoch=max_epoch, num_batches=num_batches,
                               data_name=data_name)


if __name__ == '__main__':
    submitter = SlurmSubmitter(work_dir="../", stop_on_error=True, on_local=on_local)
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
    random_seeds = [10, ]
    max_epoch = 50
    num_batches = 300

    for job in run_pretrain_ft_with_grid_search(save_dir=save_dir, random_seeds=random_seeds, max_epoch=max_epoch,
                                                num_batches=num_batches,
                                                data_name=data_name, mi_weights=[1], cc_weights=[0, 0.1, 0.2, 0.5, 1],
                                                consistency_weights=[0, 0.1, 0.5],
                                                include_baseline=True,
                                                paddings=[0, 1, 2], lamdas=[1, 1.5, 2],
                                                powers=[1, 0.75, 0.5]
                                                ):
        submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))

    for job in run_pretrain_ft_with_grid_search(save_dir=save_dir, random_seeds=random_seeds, max_epoch=max_epoch,
                                                num_batches=num_batches,
                                                data_name=data_name, mi_weights=[0, 0.1, 0.2, 0.5, 1], cc_weights=[1],
                                                consistency_weights=[0, 0.1, 0.5],
                                                include_baseline=False,
                                                paddings=[0, 1, 2], lamdas=[1, 1.5, 2],
                                                powers=[1, 0.75, 0.5]
                                                ):
        submitter.submit(" && \n ".join(job), force_show=force_show, time=4, account=next(account))

    for job in run_semi_regularize_with_grid_search(save_dir=os.path.join(save_dir, "semi"), random_seeds=random_seeds,
                                                    max_epoch=max_epoch, num_batches=num_batches,
                                                    data_name=data_name,
                                                    mi_weights=[0, 0.001, 0.01, 0.002, 0.1, 0.5, 1],
                                                    cc_weights=[0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, ],
                                                    consistency_weights=[0, 0.1, 0.5],
                                                    include_baseline=True,
                                                    paddings=[0, 1, 2], lamdas=[1, 1.5, 2],
                                                    powers=[1, 0.75, 0.5]
                                                    ):
        submitter.submit(" && \n ".join(job), force_show=force_show, time=6, account=next(account))
