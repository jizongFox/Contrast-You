#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

max_epoch=100
num_batches=256
declare -a jobs=(
"python main.py Trainer.name=fs Trainer.save_dir=fs_no_contrast Data.use_contrast=False Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} "
"python main.py Trainer.name=fs Trainer.save_dir=fs_contrast Data.use_contrast=True Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} "

"python main.py Trainer.name=semi Trainer.save_dir=semi_1.0_contrast Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=True Data.labeled_data_ratio=1.0 Data.unlabeled_data_ratio=0.0 "
"python main.py Trainer.name=semi Trainer.save_dir=semi_1.0 Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=False Data.labeled_data_ratio=1.0 Data.unlabeled_data_ratio=0.0"

"python main.py Trainer.name=semi Trainer.save_dir=semi_0.9_contrast Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=True Data.labeled_data_ratio=0.9 Data.unlabeled_data_ratio=0.1 "
"python main.py Trainer.name=semi Trainer.save_dir=semi_0.9 Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=False Data.labeled_data_ratio=0.9 Data.unlabeled_data_ratio=0.1"
"python main.py Trainer.name=semi Trainer.save_dir=semi_0.95_contrast Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=True Data.labeled_data_ratio=0.95 Data.unlabeled_data_ratio=0.05 "
"python main.py Trainer.name=semi Trainer.save_dir=semi_0.95 Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=False Data.labeled_data_ratio=0.95 Data.unlabeled_data_ratio=0.05"
#
"python main.py Trainer.name=semi Trainer.save_dir=semi_0.1_contrast Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=True Data.labeled_data_ratio=0.1 Data.unlabeled_data_ratio=0.9 "
"python main.py Trainer.name=semi Trainer.save_dir=semi_0.1 Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=False Data.labeled_data_ratio=0.1 Data.unlabeled_data_ratio=0.9"
#
"python main.py Trainer.name=semi Trainer.save_dir=semi_0.05_contrast Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=True Data.labeled_data_ratio=0.05 Data.unlabeled_data_ratio=0.95 "
"python main.py Trainer.name=semi Trainer.save_dir=semi_0.05 Trainer.max_epoch=${max_epoch} Trainer.num_batches=${num_batches} Data.use_contrast=False Data.labeled_data_ratio=0.05 Data.unlabeled_data_ratio=0.95"
)
# just using 0 and 1 gpus for those jobs
gpuqueue "${jobs[@]}" --available_gpus 0 1