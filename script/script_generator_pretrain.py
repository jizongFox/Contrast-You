import os
from itertools import cycle

from contrastyou import PROJECT_PATH
from contrastyou.configure import dictionary_merge_by_hierachy
from contrastyou.configure.yaml_parser import yaml_load, yaml_write
from contrastyou.submitter import CCSubmitter as JobSubmiter
from script import utils
from script.utils import TEMP_DIR, grid_search, PretrainScriptGenerator, move_dataset
from semi_seg import __accounts, num_batches_zoo, ft_max_epoch_zoo, pre_max_epoch_zoo

account = cycle(__accounts)
opt_hook_path = {"infonce": "config/hooks/infonce.yaml",
                 "spinfonce": "config/hooks/spinfonce.yaml"}


class PretrainInfoNCEScriptGenerator(PretrainScriptGenerator):

    def __init__(self, *, data_name, num_batches, save_dir, pre_max_epoch, ft_max_epoch) -> None:
        super().__init__(data_name=data_name, num_batches=num_batches, save_dir=save_dir,
                         pre_max_epoch=pre_max_epoch, ft_max_epoch=ft_max_epoch)

        self.hook_config = yaml_load(PROJECT_PATH + "/" + opt_hook_path[self.get_hook_name()])

    def get_hook_name(self):
        return "infonce"

    def get_hook_params(self, weight, contrast_on):
        return {"InfonceParams": {"weights": weight,
                                  "contrast_ons": contrast_on}}

    def grid_search_on(self, *, seed, **kwargs):
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



class PretrainSPInfoNCEScriptGenerator(PretrainInfoNCEScriptGenerator):

    def get_hook_name(self):
        return "spinfonce"

    def get_hook_params(self, weight, contrast_on, begin_values, end_values, mode, correct_grad):
        return {"SPInfonceParams": {"weights": weight,
                                    "contrast_ons": contrast_on,
                                    "begin_values": begin_values,
                                    "end_values": end_values,
                                    "mode": mode,
                                    "correct_grad": correct_grad
                                    }}


if __name__ == '__main__':
    submittor = JobSubmiter(work_dir="../", stop_on_error=True)
    submittor.configure_environment([
        "module load python/3.8.2 ",
        f"source ~/venv/bin/activate ",
        'if [ $(which python) == "/usr/bin/python" ]',
        "then",
        "exit 1314520",
        "fi",
        "export OMP_NUM_THREADS=1",
        "export PYTHONOPTIMIZE=1",
        "export PYTHONWARNINGS=ignore ",
        "export CUBLAS_WORKSPACE_CONFIG=:16:8 ",
        move_dataset()
    ])
    submittor.configure_sbatch(account=None)

    seed = [10, 20, 30]
    data_name = "mmwhsct"
    save_dir = f"0526/{data_name}"
    num_batches = num_batches_zoo[data_name]
    pre_max_epoch = pre_max_epoch_zoo[data_name]
    ft_max_epoch = ft_max_epoch_zoo[data_name]

    baseline_generator = PretrainInfoNCEScriptGenerator(data_name=data_name, num_batches=num_batches,
                                                        save_dir=f"{save_dir}/baseline",
                                                        pre_max_epoch=0, ft_max_epoch=ft_max_epoch)
    jobs = baseline_generator.grid_search_on(weight=1, contrast_on=("",), seed=seed)

    for j in jobs:
        submittor.account = next(account)
        submittor.submit(j, on_local=True, )

    pretrain_generator = PretrainInfoNCEScriptGenerator(data_name=data_name, num_batches=num_batches,
                                                        save_dir=f"{save_dir}/infonce",
                                                        pre_max_epoch=pre_max_epoch,
                                                        ft_max_epoch=ft_max_epoch)
    jobs = pretrain_generator.grid_search_on(weight=1, contrast_on=("partition", "patient", "self"), seed=seed)
    print(jobs)

    for j in jobs:
        submittor.account = next(account)
        submittor.run(j)

    pretrain_sp_generator = PretrainSPInfoNCEScriptGenerator(data_name=data_name, num_batches=num_batches,
                                                             save_dir=f"{data_name}/spinfonce",
                                                             pre_max_epoch=pre_max_epoch, ft_max_epoch=ft_max_epoch)
    jobs = pretrain_sp_generator.grid_search_on(weight=1, contrast_on=("partition", "self", "patient"),
                                                begin_values=(1, 2, 3, 4), end_values=(20, 30, 40, 50, 60), mode="soft",
                                                correct_grad=[False], seed=seed)
    for j in jobs:
        submittor.account = next(account)
        submittor.run(j)
