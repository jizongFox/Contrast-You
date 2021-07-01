import argparse
import os
from itertools import cycle

from contrastyou import CONFIG_PATH, on_cc, git_hash
from contrastyou.configure import dictionary_merge_by_hierachy
from contrastyou.configure.yaml_parser import yaml_load, yaml_write
from contrastyou.submitter import SlurmSubmitter as JobSubmiter
from script import utils
from script.utils import TEMP_DIR, grid_search, BaselineGenerator, \
    move_dataset
from semi_seg import __accounts, num_batches_zoo, ft_max_epoch_zoo, ratio_zoo

account = cycle(__accounts)


class MeanTeacherScriptGenerator(BaselineGenerator):

    def __init__(self, *, data_name, num_batches, max_epoch, save_dir, model_checkpoint=None) -> None:
        super().__init__(data_name=data_name, num_batches=num_batches, max_epoch=max_epoch, save_dir=save_dir,
                         model_checkpoint=model_checkpoint)

        self.hook_config = yaml_load(os.path.join(CONFIG_PATH, "hooks", "mt.yaml"))

    def get_hook_params(self, weight, two_stage, ):
        return {
            "MeanTeacherParameters":
                {"weight": weight},
            "Trainer":
                {"two_stage": two_stage}
        }

    def generate_single_script(self, save_dir, labeled_scan_num, seed, hook_path):
        from semi_seg import ft_lr_zooms
        ft_lr = ft_lr_zooms[self._data_name]

        return f"python main.py Trainer.name=semi  Trainer.save_dir={save_dir} " \
               f" Optim.lr={ft_lr:.7f} RandomSeed={str(seed)} Data.labeled_scan_num={int(labeled_scan_num)} " \
               f" {' '.join(self.conditions)} " \
               f" --opt-path {hook_path}"

    def grid_search_on(self, *, seed, **kwargs):
        jobs = []

        labeled_scan_list = ratio_zoo[self._data_name][:-1] if len(ratio_zoo[self._data_name]) > 1 else ratio_zoo[
            self._data_name]

        for param in grid_search(**{**kwargs, **{"seed": seed}}):
            random_seed = param.pop("seed")
            hook_params = self.get_hook_params(**param)
            sub_save_dir = self._get_hyper_param_string(**param)
            merged_config = dictionary_merge_by_hierachy(self.hook_config, hook_params)
            config_path = yaml_write(merged_config, save_dir=TEMP_DIR, save_name=utils.random_string() + ".yaml")
            true_save_dir = os.path.join(self._save_dir, "Seed_" + str(random_seed), sub_save_dir)

            job = " && ".join(
                [self.generate_single_script(save_dir=os.path.join(true_save_dir, "tra", f"labeled_scan_{l:02d}"),
                                             seed=random_seed, hook_path=config_path, labeled_scan_num=l)
                 for l in labeled_scan_list])

            jobs.append(job)
        return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser("adv method")
    parser.add_argument("--data-name", required=True, type=str, help="dataset_name",
                        choices=["acdc", "prostate", "mmwhsct"])
    parser.add_argument("--save_dir", required=True, type=str, help="save_dir")
    parser.add_argument("--force-show", action="store_true")

    args = parser.parse_args()

    submittor = JobSubmiter(work_dir="../", stop_on_error=False, on_local=not on_cc())
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
        "export LOGURU_LEVEL=TRACE",
        "echo $(pwd)",
        move_dataset(),
        "python -c 'import torch; print(torch.randn(1,1,1,1,device=\"cuda\"))'",
        "nvidia-smi"
    ])
    submittor.configure_sbatch(mem=48)
    seed = [10, 20, 30]
    data_name = args.data_name
    save_dir = f"{args.save_dir}/mt/hash_{git_hash}/{data_name}"
    num_batches = num_batches_zoo[data_name]
    max_epoch = ft_max_epoch_zoo[data_name]
    force_show = args.force_show

    script_generator = MeanTeacherScriptGenerator(data_name=data_name, save_dir=os.path.join(save_dir, "mt"),
                                                  num_batches=num_batches,
                                                  max_epoch=max_epoch)

    jobs = script_generator.grid_search_on(seed=seed, two_stage=[True],
                                           weight=[0, 0.001, 0.01, 0.1, 1, 5, 10, 20])

    for j in jobs:
        submittor.submit(j, account=next(account), force_show=force_show, time=8)
