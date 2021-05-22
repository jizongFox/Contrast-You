import os
from itertools import cycle

from deepclustering2.cchelper import JobSubmiter

from contrastyou import PROJECT_PATH
from contrastyou.config import dictionary_merge_by_hierachy
from script import utils
from script.utils import TEMP_DIR, yaml_load, write_yaml, grid_search, PretrainScriptGenerator, move_dataset
from semi_seg import __accounts

account = cycle(__accounts)


class PretrainInfoNCEScriptGenerator(PretrainScriptGenerator):
    opt_hook_path = {"infonce": "config/hooks/infonce.yaml",
                     "spinfonce": "config/hooks/spinfonce.yaml"}

    def __init__(self, *, data_name, num_batches, save_dir, pre_max_epoch, ft_max_epoch) -> None:
        super().__init__(data_name=data_name, num_batches=num_batches, save_dir=save_dir,
                         pre_max_epoch=pre_max_epoch, ft_max_epoch=ft_max_epoch)

        self.hook_config = yaml_load(PROJECT_PATH + "/" + self.opt_hook_path[self.get_hook_name()])

    def get_hook_name(self):
        return "infonce"

    def get_hook_params(self, weight, contrast_on):
        return {"InfonceParams": {"weights": weight,
                                  "contrast_ons": contrast_on}}

    def grid_search_on(self, seed: int, **kwargs):
        jobs = []
        for param in grid_search(**{**kwargs, **{"seed": seed}}):
            random_seed = param.pop("seed")
            hook_params = self.get_hook_params(**param)
            sub_save_dir = self.get_hyparam_string(**param)
            merged_config = dictionary_merge_by_hierachy(self.hook_config, hook_params)
            config_path = write_yaml(merged_config, save_dir=TEMP_DIR, save_name=utils.random_string() + ".yaml")
            job = self.generate_single_script(os.path.join(self._save_dir, "Seed_" + str(random_seed), sub_save_dir),
                                              seed=random_seed, hook_path=config_path)
            jobs.append(job)
        return jobs


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
    submitor = JobSubmiter(on_local=True, project_path="../")
    submitor.prepare_env([
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
    num_batches = 2
    pre_max_epoch = 2
    ft_max_epoch = 2
    seed = [10, 20, 30]

    baseline_generator = PretrainInfoNCEScriptGenerator(data_name="acdc", num_batches=num_batches,
                                                        save_dir="test_script/baseline",
                                                        pre_max_epoch=0, ft_max_epoch=ft_max_epoch)
    jobs = baseline_generator.grid_search_on(weight=1, contrast_on=("",), seed=seed)

    for j in jobs:
        submitor.account = next(account)
        submitor.run(j)

    pretrain_generator = PretrainInfoNCEScriptGenerator(data_name="acdc", num_batches=num_batches,
                                                        save_dir="test_script/infonce",
                                                        pre_max_epoch=pre_max_epoch,
                                                        ft_max_epoch=ft_max_epoch)
    jobs = pretrain_generator.grid_search_on(weight=1, contrast_on=("partition", "cycle", "self"), seed=seed)
    print(jobs)

    for j in jobs:
        submitor.account = next(account)
        submitor.run(j)

    pretrain_sp_generator = PretrainSPInfoNCEScriptGenerator(data_name="acdc", num_batches=100, save_dir="test_script",
                                                             pre_max_epoch=10, ft_max_epoch=10)
    jobs = pretrain_sp_generator.grid_search_on(weight=1, contrast_on=("partition", "cycle", "self", "patient"),
                                                begin_values=(1, 2, 3, 4), end_values=(20, 30, 40, 50, 60), mode="soft",
                                                correct_grad=[True, False], seed=seed)
    for j in jobs:
        submitor.account = next(account)
        submitor.run(j)
