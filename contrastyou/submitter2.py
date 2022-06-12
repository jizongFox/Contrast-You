import os
import subprocess
from dataclasses import dataclass
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


@dataclass()
class Config:
    job_script: str
    job_name: str = "default_job_name"
    account: str = "rrg-mpederso"
    time: int = 5
    cpu_per_task: int = 6
    mem: int = 16
    gres: str = "gpu:1"
    nodes: int = 1
    email: str = "jizong.peng.1@etsmtl.net"

    # def ___repr__(self):
    #     # should look like this:
    #     # JobEnvironment(job_id=17015819, hostname=learnfair0218, local_rank=2(3), node=1(2), global_rank=5(6))
    #     info = [f"{n}={getattr(self, n)}" for n in
    #             ("worker_dir", "account", "time_hours", "cpu_per_task", "gres", "nodes", "email")]
    #
    #     breakpoint()
    #     info_str = ", ".join(info)
    #     return f"JobEnvironment({info_str})"

    def to_script(self) -> str:
        prefix = self._create_cc_sbatch_prefix(
            job_name=self.job_name, nodes=self.nodes, gres=self.gres, cpus_per_task=self.cpu_per_task, mem=self.mem,
            mail_user=self.email, account=self.account, time=self.time
        )
        return prefix + "\n" * 2 + self.job_script

    @staticmethod
    def _create_cc_sbatch_prefix(*, job_name="default_job_name", nodes=1, gres="gpu:1",
                                 cpus_per_task=12, mem: int = 16, mail_user="jizong.peng.1@etsmtl.net",
                                 account="rrg-mpederso", time: int = 4
                                 ) -> str:
        return (
            f"#!/bin/bash \n"
            f"#SBATCH --account={account} \n"
            f"#SBATCH --time={time}:0:0 \n"
            f"#SBATCH --cpus-per-task={cpus_per_task} \n"
            f"#SBATCH --gres={gres} \n"
            f"#SBATCH --job-name={job_name} \n"
            f"#SBATCH --nodes={nodes} \n"
            f"#SBATCH --mem={mem}000M \n"
            f"#SBATCH --mail-user={mail_user} \n"
            f"#SBATCH --mail-type=FAIL \n"
        )


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


class SlurmSubmitter(AbstractSubmitter):
    cc_default_accounts = cycle(["rrg-mpederso", "def-mpederso"])

    def __init__(self, stop_on_error=False, verbose=True, dry_run: bool = False, on_local: bool = False,
                 remove_script: bool = True) -> None:
        """
        :param stop_on_error: if True, raise SubmitError when the job fails
        :param verbose: if True, print the job script
        :param dry_run: if True, do not submit the job
        :param on_local: if True, run the job on local machine
        :param remove_script: if True, remove the job script after submission
        """
        self._stop_on_error = stop_on_error
        self._verbose = verbose
        self._dry_run = dry_run
        self._on_local = on_local
        self._remove_script = remove_script

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

    def submit(self, *command_sequence: str, account: str = None, remove_script: bool = None, on_local: bool = None,
               **kwargs) -> None:

        for current_cmd in command_sequence:
            self._submit_single_job(current_cmd, account=account, remove_script=remove_script, on_local=on_local,
                                    **kwargs)

    def _submit_single_job(self, command: str, **kwargs):
        # move to the work_dir
        set_workdir_script = f"cd {self.absolute_work_dir}"

        # set environment variables
        set_env_script = ""
        if self._env:
            set_env_script = "\n".join([f"export {k}={str(v)}" for k, v in self._env.items()])

        on_local = kwargs.pop("on_local") or self._on_local
        account = kwargs.pop("account") or next(self.cc_default_accounts)
        remove_script = kwargs.pop("remove_script") or self._remove_script

        self.update_sbatch_params(account=account, **kwargs)

        prepare_script = "\n".join(self._prepare_scripts)

        full_script = "\n".join([set_workdir_script, set_env_script, prepare_script, command])

        script_config = Config(**self._sbatch_params, job_script=full_script)

        if self._verbose or self._dry_run:
            print(colored(script_config.to_script(), "green"))
        if self._dry_run:
            return
        code = self._write_run_remove(script_config.to_script(), on_local=on_local, remove_sh_script=remove_script)
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
