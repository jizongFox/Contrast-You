from itertools import cycle

from gpu_queue import JobSubmitter
from deepclustering2.cchelper import JobSubmiter as CCJobsubmitter
main_save_dir = "cmt_version1"

job_array = [
    f"OMP_NUM_THREADS=1 python -O  contrastive_mt.py -r 0 -c 0 --save_dir={main_save_dir}/ps",
    f"OMP_NUM_THREADS=1 python -O  contrastive_mt.py -r 1 -c 0 --save_dir={main_save_dir}/mt/1",
    f"OMP_NUM_THREADS=1 python -O  contrastive_mt.py -r 5 -c 0 --save_dir={main_save_dir}/mt/5",
    f"OMP_NUM_THREADS=1 python -O  contrastive_mt.py -r 10 -c 0 --save_dir={main_save_dir}/mt/10",
    f"OMP_NUM_THREADS=1 python -O  contrastive_mt.py -r 15 -c 0 --save_dir={main_save_dir}/mt/15",

    f"OMP_NUM_THREADS=1 python -O  contrastive_mt.py -r 15 -c 0 --save_dir={main_save_dir}/cmt/15/0",
    f"OMP_NUM_THREADS=1 python -O  contrastive_mt.py -r 15 -c 0.01 --save_dir={main_save_dir}/cmt/15/0.01",
    f"OMP_NUM_THREADS=1 python -O  contrastive_mt.py -r 15 -c 0.1 --save_dir={main_save_dir}/cmt/15/0.1",
    f"OMP_NUM_THREADS=1 python -O  contrastive_mt.py -r 15 -c 1 --save_dir={main_save_dir}/cmt/15/1",
    f"OMP_NUM_THREADS=1 python -O  contrastive_mt.py -r 15 -c 2 --save_dir={main_save_dir}/cmt/15/2",
    f"OMP_NUM_THREADS=1 python -O  contrastive_mt.py -r 15 -c 3 --save_dir={main_save_dir}/cmt/15/3",
    f"OMP_NUM_THREADS=1 python -O  contrastive_mt.py -r 15 -c 4 --save_dir={main_save_dir}/cmt/15/4",
]

submitter = JobSubmitter(job_array=job_array, available_gpus=[0,1])
submitter.submit_jobs()

#accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])
#jobsubmiter = CCJobsubmitter(project_path="./", on_local=True, time=8)
#for j in job_array:
#    jobsubmiter.prepare_env(["source ../venv/bin/activate ", "export OMP_NUM_THREADS=1", ])
#    jobsubmiter.account = next(accounts)
#    jobsubmiter.run(j)
