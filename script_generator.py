import os
from itertools import product

from deepclustering2.cchelper import JobSubmiter

from contrastyou.config import dictionary2string
from semi_seg import pre_lr_zooms, ft_lr_zooms, data2input_dim, data2class_numbers, ratio_zoo
from semi_seg.scripts.helper import run_jobs


def create_fine_tune_script(*, data_name, max_epoch, num_batches, lr, save_dir, seed=10, model_checkpoint=None):
    input_dim = data2input_dim[data_name]
    num_classes = data2class_numbers[data_name]
    labeled_ratios = ratio_zoo[data_name]

    return [f" python main.py Trainer.name=ft Optim.lr={lr:.7f} Data.name={data_name} RandomSeed={seed} "
            f" Trainer.save_dir={save_dir}/tra_{ratio} Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches}"
            f" Arch.input_dim={input_dim} Arch.num_classes={num_classes} " \
            f" Arch.checkpoint={model_checkpoint if model_checkpoint else 'null'} "
            for ratio in labeled_ratios]


def create_pretrain_script(*, data_name, max_epoch, num_batches, lr, save_dir, seed=10, hook_name, **hook_params):
    input_dim = data2input_dim[data_name]
    num_classes = data2class_numbers[data_name]
    opt_hook_path = {"infonce": "config/hooks/infonce.yaml",
                     "spinfonce": "config/hooks/spinfonce.yaml"}[hook_name]

    return f" python main.py Trainer.name=pretrain Optim.lr={lr:.7f} Data.name={data_name} RandomSeed={seed} " \
           f" Trainer.save_dir={save_dir} Trainer.max_epoch={max_epoch} Trainer.num_batches={num_batches}" \
           f" Arch.input_dim={input_dim} Arch.num_classes={num_classes} " \
           f" {dictionary2string(hook_params)}" \
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
                                                   lr=ft_lr, save_dir=save_dir,
                                                   model_checkpoint=os.path.join(save_dir, "pre", "last.pth"))
    if len(env) > 0:
        env_str = "\n".join(env)
        return env_str + "\n" + "\n ".join([pretrain_script, *finetune_script_list])
    return "\n ".join([pretrain_script, *finetune_script_list])


if __name__ == '__main__':
    def create_infonce_hooks(*, contrast_ons=["partition", "cycle", "patient"], weight=[1], save_dir: str):
        hook_params = lambda contrast_on, weight: {"InfonceParams":
                                                       {"feature_names": "Conv5",
                                                        "weights": weight,
                                                        "contrast_ons": contrast_on}}

        for c, w in product(contrast_ons, weight):
            _save_dir = save_dir + "/" + "/".join([f"contrast_on_{c}", f"w_{w}"])
            yield "infonce", hook_params(c, w), _save_dir


    def create_sp_infonce_hooks(*, contrast_ons=["partition", "cycle", "patient"], weight=[1], begin_values=[1, 2, 3],
                                end_values=[10, 20, 30, 40], mode=["soft"], p=[0.5], correct_grad=[True, False],
                                save_dir: str
                                ):

        hook_params = lambda contrast_on, weight, b, e, m, p, c: \
            {'SPInfonceParams': {'feature_names': 'Conv5',
                                 'weights': weight,
                                 'contrast_ons': contrast_on,
                                 'begin_values': b,
                                 'end_values': e,
                                 'mode': m,
                                 'p': p,
                                 'correct_grad': c}}

        for c, w, b, e, p, m, cg in product(contrast_ons, weight, begin_values, end_values, p, mode, correct_grad):
            _save_dir = save_dir + "/" + "/".join(
                [f"contrast_on_{c}", f"w_{w}", f"b_{b}_e_{e}", f"p_{p}", f"cor_grad_{cg}"])
            yield "spinfonce", hook_params(c, w, b, e, p, m, cg), _save_dir


    pre_max_epoch = 80
    ft_max_epoch = 80
    num_batches = 200
    save_dir = "test"

    env = ["set -e", "export PYTHONOPTIMIZE=1", "export LOGURU_LEVEL=TRACE", "export OMP_NUM_THREADS=1",
           "export CUBLAS_WORKSPACE_CONFIG=:4096:8"]
    base_job = create_pretrain_fine_tune_pipeline(data_name="acdc", pre_max_epoch=0,
                                                  ft_max_epoch=ft_max_epoch,
                                                  num_batches=num_batches,
                                                  env=env, save_dir=os.path.join(save_dir, "baseline"),
                                                  hook_name="infonce", )
    info_jobs = []
    for h_name, h_params, dir in create_infonce_hooks(save_dir="tra_infonce"):
        info_job = create_pretrain_fine_tune_pipeline(data_name="acdc", pre_max_epoch=pre_max_epoch,
                                                      ft_max_epoch=ft_max_epoch,
                                                      num_batches=num_batches,
                                                      env=env, save_dir=os.path.join(save_dir, dir),
                                                      hook_name=h_name, **h_params)
        info_jobs.append(info_job)

    sp_jobs = []
    for h_name, h_params, dir in create_sp_infonce_hooks(save_dir="sp_infonce"):
        sp_job = create_pretrain_fine_tune_pipeline(data_name="acdc", pre_max_epoch=pre_max_epoch,
                                                    ft_max_epoch=ft_max_epoch, num_batches=num_batches,
                                                    env=env, save_dir=os.path.join(save_dir, dir), hook_name=h_name, **h_params)
        sp_jobs.append(sp_job)

    job_submiter = JobSubmiter(project_path="./", on_local=False, time=4, mem=20)

    run_jobs(job_submiter, job_array, args, )


