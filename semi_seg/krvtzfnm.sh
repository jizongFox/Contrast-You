#!/bin/bash 
#SBATCH --time=0-4:00 
#SBATCH --account=def-mpederso 
#SBATCH --cpus-per-task=6 
#SBATCH --gres=gpu:1 
#SBATCH --account=def-mpederso 
#SBATCH --job-name=default_jobname 
#SBATCH --nodes=1 
#SBATCH --mem=16000M 
#SBATCH --mail-user=jizong.peng.1@etsmtl.net 
#SBATCH --mail-type=FAIL 

cd /home/jizong/Workspace/Contrast-You/semi_seg
source ../venv/bin/activate 
export OMP_NUM_THREADS=1
export PYTHONOPTIMIZE=1
export CUBLAS_WORKSPACE_CONFIG=:16:8 
python main_infonce.py Trainer.name=infoncepretrain  Data.name=acdc Trainer.num_batches=20  Arch.num_classes=4  RandomSeed=10  Optim.pre_lr=null Optim.ft_lr=null Trainer.pre_max_epoch=45 Trainer.ft_max_epoch=20  ContrastiveLoaderParams.group_sample_num=3 ProjectorParams.GlobalParams.feature_names=[Conv5] ProjectorParams.GlobalParams.feature_importance=[1.0] ProjectorParams.LossParams.contrast_on=[partition] Trainer.monitor=true  ProjectorParams.LossParams.begin_value=[4.0] ProjectorParams.LossParams.end_value=[14.0] ProjectorParams.LossParams.weight_update=[hard] ProjectorParams.LossParams.type=[square]  Trainer.save_dir=fdasjklfd/githash_fa9f9ee/acdc/random_seed_10/sample_num_3//global_Conv5_1.0/contrast_on_partition/self-paced/method_hard/loss_params*4.0_14.0/type_square  --opt_config_path ../config/specific/pretrain.yaml ../config/specific/selfpaced_infonce.yaml