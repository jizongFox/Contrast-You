import collections
import os
from collections.abc import Iterable

from semi_seg import pre_lr_zooms, ft_lr_zooms, data2input_dim, data2class_numbers, ratio_zoo


def tostring(item):
    if isinstance(item, (float,)):
        return f"{item:.7f}"
    return str(item)


def dict2string(dictionary, parent_name_list=None, item_list=None):
    if parent_name_list is None:
        parent_name_list = []
    if item_list is None:
        item_list = []
    for k, v in dictionary.items():
        if isinstance(v, collections.abc.Mapping):
            dict2string(v, parent_name_list=parent_name_list + [k], item_list=item_list)
        elif isinstance(v, Iterable) and (not isinstance(v, str)):
            current_item = ".".join(parent_name_list) + f".{k}=[{','.join([tostring(x) for x in v])}]"
            item_list.append(current_item)
        else:
            current_item = ".".join(parent_name_list) + f".{k}={tostring(v)}"
            item_list.append(current_item)
    return " ".join(item_list)


def create_fine_tune_script(*, data_name, max_epoch, num_batches, lr, save_dir, seed=10):
    input_dim = data2input_dim[data_name]
    num_classes = data2class_numbers[data_name]
    labeled_ratios = ratio_zoo[data_name]

    return [f" python main.py Trainer.name=ft Optim.lr={lr:.7f} Data.name={data_name} RandomSeed={seed} "
            f" Trainer.save_dir={save_dir}/tra_{ratio} Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches}"
            f" Arch.input_dim={input_dim} Arch.num_classes={num_classes} " for ratio in labeled_ratios]


def create_pretrain_script(*, data_name, max_epoch, num_batches, lr, save_dir, seed=10, hook_name, **hook_params):
    input_dim = data2input_dim[data_name]
    num_classes = data2class_numbers[data_name]
    opt_hook_path = {"infonce": "config/hooks/infonce.yaml",
                     "spinfonce": "config/hooks/spinfonce.yaml"}[hook_name]

    return f" python main.py Trainer.name=pretrain Optim.lr={lr:.7f} Data.name={data_name} RandomSeed={seed} " \
           f" Trainer.save_dir={save_dir} Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches}" \
           f" Arch.input_dim={input_dim} Arch.num_classes={num_classes} " \
           f" {dict2string(hook_params)}" \
           f" --opt-path " \
           f" config/pretrain.yaml " \
           f" {opt_hook_path} "


def create_pretrain_fine_tune_pipeline(*, data_name, pre_max_epoch, ft_max_epoch, num_batches, save_dir, hook_name,
                                       env=[],
                                       **hook_params, ):
    pre_lr, ft_lr = pre_lr_zooms[data_name], ft_lr_zooms[data_name]
    pretrain_script = create_pretrain_script(data_name=data_name, max_epoch=pre_max_epoch, num_batches=num_batches,
                                             lr=pre_lr, save_dir=os.path.join(save_dir, "pre"), hook_name=hook_name,
                                             **hook_params)
    finetune_script_list = create_fine_tune_script(data_name=data_name, max_epoch=ft_max_epoch, num_batches=num_batches,
                                                   lr=ft_lr, save_dir=save_dir)
    env_str = "\n".join(env)

    return env_str + "\n" + "\n ".join([pretrain_script, *finetune_script_list])


if __name__ == '__main__':
    hook_params = {
        "InfonceParams": {"feature_names": ["Conv5", "Conv4"], "weights": [1, 0.9, ], "contrast_ons": "partition"}
    }
    env = ["set -e", "export PYTHONOPTIMIZE=1", "export LOGURU_LEVEL=TRACE", "OMP_NUM_THREAS=1",
           "export CUBLAS_WORKSPACE_CONFIG=:4096:8"]
    jobs = create_pretrain_fine_tune_pipeline(data_name="acdc", pre_max_epoch=10, ft_max_epoch=5, num_batches=20,
                                              env=env, save_dir="tmp", hook_name="infonce", **hook_params)

    import subprocess

    subprocess.call(jobs, shell=True)
