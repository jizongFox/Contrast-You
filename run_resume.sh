#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
python main.py Trainer.name=fs Trainer.save_dir=fs Trainer.max_epoch=50 Trainer.num_batches=512
python main.py --config_path=runs/fs Trainer.save_dir=resume Trainer.max_epoch=100 Trainer.checkpoint=runs/fs
