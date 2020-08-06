from itertools import cycle

from gpu_queue import JobSubmitter
from deepclustering2.cchelper import JobSubmiter as CCJobsubmitter
main_save_dir = "cmt_version2"

job_array = [
    f"python -O  contrastive_mt.py -r 0  -c 0 --update_bn --save_dir={main_save_dir}/ps/bn",
    f"python -O  contrastive_mt.py -r 1  -c 0 --update_bn --save_dir={main_save_dir}/mt/bn/1",
    f"python -O  contrastive_mt.py -r 5  -c 0 --update_bn --save_dir={main_save_dir}/mt/bn/5",
    f"python -O  contrastive_mt.py -r 10 -c 0 --update_bn --save_dir={main_save_dir}/mt/bn/10",
    f"python -O  contrastive_mt.py -r 15 -c 0 --update_bn --save_dir={main_save_dir}/mt/bn/15",

    f"python -O  contrastive_mt.py -r 0  -c 0  --save_dir={main_save_dir}/ps/no_bn",
    f"python -O  contrastive_mt.py -r 1  -c 0  --save_dir={main_save_dir}/mt/no_bn/1",
    f"python -O  contrastive_mt.py -r 5  -c 0  --save_dir={main_save_dir}/mt/no_bn/5",
    f"python -O  contrastive_mt.py -r 10 -c 0  --save_dir={main_save_dir}/mt/no_bn/10",
    f"python -O  contrastive_mt.py -r 15 -c 0  --save_dir={main_save_dir}/mt/n0_bn/15",

    # f"python -O  contrastive_mt.py -r 1 -c 0 --save_dir={main_save_dir}/cmt/1/0",
    # f"python -O  contrastive_mt.py -r 1 -c 0.01 --save_dir={main_save_dir}/cmt/1/0.01",
    # f"python -O  contrastive_mt.py -r 1 -c 0.1 --save_dir={main_save_dir}/cmt/1/0.1",
    # f"python -O  contrastive_mt.py -r 1 -c 1 --save_dir={main_save_dir}/cmt/1/1",
    # f"python -O  contrastive_mt.py -r 1 -c 2 --save_dir={main_save_dir}/cmt/1/2",
    # f"python -O  contrastive_mt.py -r 1 -c 3 --save_dir={main_save_dir}/cmt/1/3",
    # f"python -O  contrastive_mt.py -r 1 -c 4 --save_dir={main_save_dir}/cmt/1/4",
    #
    #
    # f"python -O  contrastive_mt.py -r 5 -c 0 --save_dir={main_save_dir}/cmt/5/0",
    # f"python -O  contrastive_mt.py -r 5 -c 0.01 --save_dir={main_save_dir}/cmt/5/0.01",
    # f"python -O  contrastive_mt.py -r 5 -c 0.1 --save_dir={main_save_dir}/cmt/5/0.1",
    # f"python -O  contrastive_mt.py -r 5 -c 1 --save_dir={main_save_dir}/cmt/5/1",
    # f"python -O  contrastive_mt.py -r 5 -c 2 --save_dir={main_save_dir}/cmt/5/2",
    # f"python -O  contrastive_mt.py -r 5 -c 3 --save_dir={main_save_dir}/cmt/5/3",
    # f"python -O  contrastive_mt.py -r 5 -c 4 --save_dir={main_save_dir}/cmt/5/4",
    #
    #
    # f"python -O  contrastive_mt.py -r 10 -c 0 --save_dir={main_save_dir}/cmt/10/0",
    # f"python -O  contrastive_mt.py -r 10 -c 0.01 --save_dir={main_save_dir}/cmt/10/0.01",
    # f"python -O  contrastive_mt.py -r 10 -c 0.1 --save_dir={main_save_dir}/cmt/10/0.1",
    # f"python -O  contrastive_mt.py -r 10 -c 1 --save_dir={main_save_dir}/cmt/10/1",
    # f"python -O  contrastive_mt.py -r 10 -c 2 --save_dir={main_save_dir}/cmt/10/2",
    # f"python -O  contrastive_mt.py -r 10 -c 3 --save_dir={main_save_dir}/cmt/10/3",
    # f"python -O  contrastive_mt.py -r 10 -c 4 --save_dir={main_save_dir}/cmt/10/4",
    #
    # f"python -O  contrastive_mt.py -r 15 -c 0 --save_dir={main_save_dir}/cmt/15/0",
    # f"python -O  contrastive_mt.py -r 15 -c 0.01 --save_dir={main_save_dir}/cmt/15/0.01",
    # f"python -O  contrastive_mt.py -r 15 -c 0.1 --save_dir={main_save_dir}/cmt/15/0.1",
    # f"python -O  contrastive_mt.py -r 15 -c 1 --save_dir={main_save_dir}/cmt/15/1",
    # f"python -O  contrastive_mt.py -r 15 -c 2 --save_dir={main_save_dir}/cmt/15/2",
    # f"python -O  contrastive_mt.py -r 15 -c 3 --save_dir={main_save_dir}/cmt/15/3",
    # f"python -O  contrastive_mt.py -r 15 -c 4 --save_dir={main_save_dir}/cmt/15/4",
]

# submitter = JobSubmitter(job_array=job_array, available_gpus=[0,1])
# submitter.submit_jobs()

accounts = cycle(["def-chdesa", "def-mpederso", "rrg-mpederso"])
jobsubmiter = CCJobsubmitter(project_path="./", on_local=True, time=4)
for j in job_array:
   jobsubmiter.prepare_env(["source ../venv/bin/activate ", "export OMP_NUM_THREADS=1", ])
   jobsubmiter.account = next(accounts)
   jobsubmiter.run(j)
