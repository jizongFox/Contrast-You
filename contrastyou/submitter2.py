import os
import subprocess
from itertools import cycle
from pprint import pprint, pformat
from typing import Dict, Any, Optional, Tuple

from termcolor import colored
from typing_extensions import Protocol


class SubmitError(RuntimeError):
    pass


def randomString():
    import random
    import string
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(10))


class AbstractSubmitter(Protocol):
    def submit(self, *command_sequence: str, **kwargs) -> None:
        ...

    def set_prefix(self, prefix: str) -> None:
        ...

    def set_startpoint_path(self, startpoint_path: str) -> None:
        ...

    def set_env_params(self, **env_params: Any) -> None:
        ...

    def update_env_params(self, **env_params: Any) -> None:
        ...

    def set_sbatch_params(self, **kwargs: Any) -> None:
        ...

    def update_sbatch_params(self, **kwargs: Any) -> None:
        ...


def _create_cc_sbatch_prefix(
        *, job_name="default_job_name", nodes=1, gres="gpu:1",
        cpus_per_task=12, mem: int = 16, mail_user="jizong.peng.1@etsmtl.net", account="rrg-mpederso", time: int = 4,
        node=1
) -> str:
    return (
        f"#!/bin/bash \n"
        f"#SBATCH --account={account} \n"
        f"#SBATCH --time={time}:0:0 \n"
        f"#SBATCH --node={node} \n"
        f"#SBATCH --cpus-per-task={cpus_per_task} \n"
        f"#SBATCH --gres={gres} \n"
        f"#SBATCH --job-name={job_name} \n"
        f"#SBATCH --nodes={nodes} \n"
        f"#SBATCH --mem={mem}000M \n"
        f"#SBATCH --mail-user={mail_user} \n"
        f"#SBATCH --mail-type=FAIL \n"
    )


class SlurmSubmitter(AbstractSubmitter):
    cc_default_accounts = cycle(["rrg-mpederso", "def-mpederso"])

    def __init__(self, stop_on_error=False, verbose=True, dry_run: bool = False) -> None:
        """
        :param stop_on_error: if True, raise SubmitError when the job fails
        :param verbose: if True, print the job script
        :param dry_run: if True, do not submit the job
        """
        self._stop_on_error = stop_on_error
        self._verbose = verbose
        self._dry_run = dry_run

        self._work_dir: Optional[str] = None
        self._env: Dict[str, Any] = {}
        self._prepare_scripts: Tuple[str, ...] = ()
        self._sbatch_params: Dict[str, Any] = {}

    def set_startpoint_path(self, startpoint_path: str) -> None:
        self._work_dir = startpoint_path

    @property
    def absolute_work_dir(self) -> str:
        if self._work_dir is not None:
            return os.path.abspath(self._work_dir)
        raise SubmitError("work_dir is not set")

    def set_env_params(self, **env_params: Any) -> None:
        self._env = env_params

    def update_env_params(self, **env_params: Any) -> None:
        self._env.update(env_params)

    @property
    def env(self) -> str:
        return "" if len(self._env) == 0 else pformat(self._env)

    def set_prepare_scripts(self, *prepare_scripts: str) -> None:
        self._prepare_scripts = prepare_scripts

    @property
    def sbatch_params(self) -> str:
        return "" if len(self._sbatch_params) == 0 else pformat(self._sbatch_params)

    def set_sbatch_params(self, **kwargs: Any) -> None:
        self._sbatch_params = kwargs

    def update_sbatch_params(self, **kwargs: Any) -> None:
        self._sbatch_params.update(kwargs)

    def sbatch_prefix(self) -> str:
        return _create_cc_sbatch_prefix(**self._sbatch_params)

    def set_default_accounts(self, *accounts: str) -> None:
        self.cc_default_accounts = cycle(list(accounts))

    def submit(self, *command_sequence: str, account: str = None, remove_script: bool = True, on_local: bool,
               **kwargs) -> None:

        # move to the work_dir
        set_workdir_script = f"cd {self.absolute_work_dir}"

        # set environment variables
        set_env_script = ""
        if len(self._env) > 0:
            set_env_script = "\n".join([f"export {k}={str(v)}" for k, v in self._env.items()])

        for current_cmd in command_sequence:
            # sbatch params:
            current_account = account or next(self.cc_default_accounts)
            self.update_sbatch_params(account=current_account)
            sbatch_prefix = self.sbatch_prefix()

            prepare_script = "\n".join(self._prepare_scripts)

            full_script = "\n".join([sbatch_prefix, set_workdir_script, set_env_script, prepare_script, current_cmd])
            if self._verbose or self._dry_run:
                print(colored(full_script, "green"))
            if self._dry_run:
                continue
            code = self._write_run_remove(full_script, on_local=on_local, remove_sh_script=remove_script)
            if code == 127:
                if self._stop_on_error:
                    raise SubmitError("sbatch not found on the machine. Please run with on_local=true")
            elif code != 0:
                if self._stop_on_error:
                    raise SubmitError(code)

    def _write_run_remove(self, full_script: str, *, on_local: bool = False, remove_sh_script: bool = True) -> int:
        """
        write the script to the work_dir, run it and remove the script
        param full_script: the script to run
        param on_local: if True, run the script using bash on the local machine, else using sbatch on cluster
        param remove_sh_script: if True, remove the script after running it
        """
        random_name = f"{randomString()}.sh"
        workdir = self.absolute_work_dir
        random_bash = os.path.join(workdir, random_name)

        with open(random_bash, "w") as f:
            f.write(full_script)
        try:
            if on_local:
                code = subprocess.call(f"bash {random_bash}", shell=True)
            else:
                code = subprocess.call(f"sbatch {random_bash}", shell=True)

        finally:
            if os.path.exists(random_bash) and remove_sh_script:
                os.remove(random_bash)
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
    parser.add_argument("--single-job", required=True, type=str, help="job script")
    args = parser.parse_args()
    pprint(args)
    return args


def main():
    args = get_args()
    submitter = SlurmSubmitter(stop_on_error=True, dry_run=True, verbose=True)
    submitter.set_startpoint_path(args.work_dir)
    submitter.set_sbatch_params(account=args.account, cpus_per_task=args.cpus_per_task, mem=args.mem, gres=args.gres,
                                time=args.time, node=1)
    submitter.set_env_params(LOGURU_LOGLEVEL="trace", PYTHONOPTIMIZE=1)
    submitter.submit(args.single_job, on_local=args.on_local, remove_script=True, )


if __name__ == "__main__":
    main()
