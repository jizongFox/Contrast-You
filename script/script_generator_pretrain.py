import os
from itertools import cycle
from typing import Union, List

from contrastyou import CONFIG_PATH
from contrastyou.configure import dictionary_merge_by_hierachy
from contrastyou.configure.yaml_parser import yaml_load, yaml_write
from contrastyou.submitter import SlurmSubmitter as JobSubmiter
from script import utils
from script.utils import TEMP_DIR, grid_search, PretrainScriptGenerator, move_dataset
from semi_seg import __accounts, pre_max_epoch_zoo, num_batches_zoo, ft_max_epoch_zoo, ft_lr_zooms

"""maybe I should let this file to be as easy as possible."""
account = cycle(__accounts)

"""base script"""
"""python main_pretrain_encoder.py 
    *base*
    Trainer.save_dir:str
    Trainer.pre_max_epoch:int 
    Trainer.ft_max_epoch:int 
    Trainer.num_batches:int
    RandomSeed:int
    Data.name:str
    pre_lr:float
    ft_lr:float
    Arch.input_dim:int 
    Arch.num_classes:int
    *hook*
    SPInfonceParams.weights:float
    SPInfonceParams.contrast_on: str
    SPInfonceParams.begin_value:float
    SPInfonceParams.end_values:str
    SPInfonceParams.mode:str
    
    --opt-path ./config/pretrain.yaml  ./config/hooks/spinfonce.yaml
"""


class PretrainSPInfoNCEScriptGenerator(PretrainScriptGenerator):

    def __init__(self, *, data_name, num_batches, save_dir, pre_max_epoch, ft_max_epoch) -> None:
        super().__init__(data_name=data_name, num_batches=num_batches, save_dir=save_dir,
                         pre_max_epoch=pre_max_epoch, ft_max_epoch=ft_max_epoch)

        self.hook_config = yaml_load(os.path.join(CONFIG_PATH, "hooks", "spinfonce.yaml"))

    def get_hook_params(self, weight, contrast_on, begin_values, end_values, mode, correct_grad):
        return {"SPInfonceParams": {"weights": weight,
                                    "contrast_ons": contrast_on,
                                    "begin_values": begin_values,
                                    "end_values": end_values,
                                    "mode": mode,
                                    "correct_grad": correct_grad
                                    }}

    def grid_search_on(self, *, seed: Union[int, List[int]], **kwargs):
        jobs = []
        for param in grid_search(**{**kwargs, **{"seed": seed}}):
            random_seed = param.pop("seed")
            hook_params = self.get_hook_params(**param)
            sub_save_dir = self._get_hyper_param_string(**param)
            merged_config = dictionary_merge_by_hierachy(self.hook_config, hook_params)
            config_path = yaml_write(merged_config, save_dir=TEMP_DIR, save_name=utils.random_string() + ".yaml")
            true_save_dir = os.path.join(self._save_dir, "Seed_" + str(random_seed), sub_save_dir)
            job = self.generate_single_script(save_dir=true_save_dir,
                                              seed=random_seed, hook_path=config_path)
            jobs.append(job)
        return jobs

    def generate_single_script(self, save_dir, seed, hook_path):
        from semi_seg import pre_lr_zooms, ft_lr_zooms
        pre_lr = pre_lr_zooms[self._data_name]
        ft_lr = ft_lr_zooms[self._data_name]
        return f"python main_pretrain_encoder.py Trainer.save_dir={save_dir} " \
               f" Optim.pre_lr={pre_lr:.7f} Optim.ft_lr={ft_lr:.7f} RandomSeed={str(seed)} " \
               f" {' '.join(self.conditions)}  " \
               f" --opt-path config/pretrain.yaml {hook_path}"


if __name__ == '__main__':
    submittor = JobSubmiter(work_dir="../", stop_on_error=True, on_local=True)
    submittor.configure_environment([
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
        # "export LOGURU_LEVEL=TRACE",
        "echo $(pwd)",
        move_dataset(),
        "nvidia-smi",
        "python -c 'import torch; print(torch.randn(1,1,1,1,device=\"cuda\"))'"
    ])
    submittor.configure_sbatch(mem=48)

    seed = [10, 20, 30]
    data_name = "acdc"
    save_dir = f"contrastive_learn/{data_name}"
    num_batches = num_batches_zoo[data_name]
    pre_max_epoch = pre_max_epoch_zoo[data_name]
    ft_max_epoch = ft_max_epoch_zoo[data_name]
    lr = ft_lr_zooms[data_name]
    force_show = True
    on_local = False
    contrast_on = ["partition", "cycle", "patient", "self"] if data_name == "acdc" else ["partition", "patient", "self"]

    baseline_generator = PretrainSPInfoNCEScriptGenerator(
        data_name=data_name, num_batches=num_batches, save_dir=f"{save_dir}/baseline", pre_max_epoch=0,
        ft_max_epoch=ft_max_epoch
    )
    jobs = baseline_generator.grid_search_on(
        seed=seed, weight=0, contrast_on="", begin_values=0, end_values=0, mode="", correct_grad=False
    )

    for j in jobs:
        submittor.submit(j, on_local=on_local, account=next(account), force_show=force_show, time=8)

    infonce_generator = PretrainSPInfoNCEScriptGenerator(
        data_name=data_name, num_batches=num_batches, save_dir=f"{save_dir}/infonce", pre_max_epoch=pre_max_epoch,
        ft_max_epoch=ft_max_epoch
    )
    jobs = infonce_generator.grid_search_on(
        seed=seed, weight=1, contrast_on=contrast_on, begin_values=1e6, end_values=1e6,
        mode="hard", correct_grad=False
    )
    for j in jobs:
        submittor.submit(j, on_local=on_local, account=next(account), force_show=force_show, time=8)

    spinfonce_generator = PretrainSPInfoNCEScriptGenerator(
        data_name=data_name, num_batches=num_batches, save_dir=f"{save_dir}/spinfonce", pre_max_epoch=pre_max_epoch,
        ft_max_epoch=ft_max_epoch
    )
    jobs = spinfonce_generator.grid_search_on(
        seed=seed, weight=1, contrast_on=contrast_on,
        begin_values=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], end_values=[10, 20, 30, 40, 50, 60, 70],
        mode="soft", correct_grad=False
    )
    for j in jobs:
        submittor.submit(j, on_local=on_local, account=next(account), force_show=force_show, time=8)
