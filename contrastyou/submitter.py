import os
import subprocess
from pprint import pprint
from typing import List, Union

from termcolor import colored


def randomString():
    import random
    import string
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10))


def _create_sbatch_prefix(*, account: str, time: int = 1, job_name="default_job_name", nodes=1, gres="gpu:1",
                          cpus_per_task=6, mem: int = 16, mail_user="jizong.peng.1@etsmtl.net"):
    return (
        f"#!/bin/bash \n"
        f"#SBATCH --time=0-{time}:00 \n"
        f"#SBATCH --account={account} \n"
        f"#SBATCH --cpus-per-task={cpus_per_task} \n"
        f"#SBATCH --gres={gres} \n"
        f"#SBATCH --account={account} \n"
        f"#SBATCH --job-name={job_name} \n"
        f"#SBATCH --nodes={nodes} \n"
        f"#SBATCH --mem={mem}000M \n"
        f"#SBATCH --mail-user={mail_user} \n"
        f"#SBATCH --mail-type=FAIL \n"
    )


class CCSubmitter:
    def __init__(self, work_dir="./", stop_on_error=False, verbose=True) -> None:
        self._work_dir = work_dir
        self._env = []
        self._sbatch_kwargs = {}
        self._configure_sbatch_done = False
        self._stop_on_error = stop_on_error
        self._verbose = verbose

    def configure_sbatch(self, **kwargs):
        self._sbatch_kwargs = kwargs
        self._configure_sbatch_done = True

    def configure_environment(self, cmd_list: Union[str, List[str]] = None):
        if isinstance(cmd_list, str):
            cmd_list = [cmd_list, ]
        self._env = cmd_list

    def submit(self, job: str, on_local=False, **kwargs):

        env_script = "\n".join(self._env)
        cd_script = f"cd {os.path.abspath(self._work_dir)}"

        script = "\n".join(
            [_create_sbatch_prefix(**{**self._sbatch_kwargs, **kwargs}), env_script, cd_script, job])
        code = self._write_and_run(script, on_local, verbose=self._verbose)
        if code != 0:
            if self._stop_on_error:
                raise RuntimeError(code)

    def _write_and_run(self, full_script, on_local, verbose=False):
        random_name = randomString() + ".sh"
        workdir = os.path.abspath(self._work_dir)
        random_bash = os.path.join(workdir, random_name)

        with open(random_bash, "w") as f:
            f.write(full_script)
        try:
            if verbose:
                print(colored(full_script, "green"), "\n")
            if on_local:
                code = subprocess.call(f"bash {random_bash}", shell=True)
            else:
                code = subprocess.call(f"sbatch {random_bash}", shell=True)
        finally:
            os.remove(random_bash)
            # pass
        return code


def get_args():
    import argparse
    parser = argparse.ArgumentParser(prog="Compute Canada Submitter",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-w", "--work-dir", default="./", help="working directory")
    parser.add_argument("--on-local", default=False, action="store_true", help="running program on local")
    parser.add_argument("--verbose", default=False, action="store_true", help="verbose")
    parser.add_argument("-a", "--account", nargs="+", type=str, default=["rrg-mpederso"], help="CC account")
    parser.add_argument("-c", "--cpus_per_task", default=6, type=int, help="cpus_per_task")
    parser.add_argument("-m", "--mem", type=int, default=16, help="memory in Gb")
    parser.add_argument("-g", "--gres", type=str, default="gres:1", help="gpu resources")
    parser.add_argument("-t", "--time", type=int, default=4, help="submit time")
    parser.add_argument("--env", type=str, nargs="*", default=[], help="environment list")
    parser.add_argument("--single_job", required=True, type=str, help="job script")
    args = parser.parse_args()
    pprint(args)
    return args


def main():
    args = get_args()
    submitter = CCSubmitter(work_dir=args.work_dir)
    submitter.configure_environment(args.env)
    submitter.configure_sbatch(cpus_per_task=args.cpus_per_task, mem=args.mem, time=args.time, nodes=1)
    submitter.submit(args.single_job, account=args.account[0], on_local=args.on_local, verbose=args.verbose)


if __name__ == "__main__":
    main()
