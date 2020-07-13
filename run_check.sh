#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

wait

declare -a jobs=(
"python main.py Trainer.name=fs Trainer.save_dir=fs Trainer.max_epoch=100 Trainer.num_batches=512 "
"python main.py Trainer.name=semi Trainer.save_dir=semi_contrast Trainer.max_epoch=100 Trainer.num_batches=512 Data.use_contrast=True "
"python main.py Trainer.name=semi Trainer.save_dir=semi Trainer.max_epoch=100 Trainer.num_batches=512 Data.use_contrast=False "
)
# just using 0 and 1 gpus for those jobs
gpuqueue "${jobs[@]}" --available_gpus 0 1