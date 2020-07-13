#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

max_epoch=100
num_batches=512
declare -a jobs=(
"python main.py Trainer.name=fs Trainer.save_dir=fs Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} "
"python main.py Trainer.name=semi Trainer.save_dir=semi_0.9_contrast Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=True Data.labeled_data_ratio=0.9 Data.unlabeled_data_ratio=0.1 "
"python main.py Trainer.name=semi Trainer.save_dir=semi_0.9 Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=False Data.labeled_data_ratio=0.9 Data.unlabeled_data_ratio=0.1"
"python main.py Trainer.name=semi Trainer.save_dir=semi_0.1_contrast Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=True Data.labeled_data_ratio=0.1 Data.unlabeled_data_ratio=0.9 "
"python main.py Trainer.name=semi Trainer.save_dir=semi_0.1 Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=False Data.labeled_data_ratio=0.1 Data.unlabeled_data_ratio=0.9"
)
# just using 0 and 1 gpus for those jobs
gpuqueue "${jobs[@]}" --available_gpus 0 1