#!/usr/bin/env bash
set -e
main_savedir=0809/contrastive
batchsize=200
# test basic partition
python run_script.py -l 0.01 -n contrast -b ${batchsize} -o partition --save_dir=${main_savedir}/test_partition
python run_script.py -l 0.01 -n contrast -b ${batchsize} -o patient --save_dir=${main_savedir}/test_partition
python run_script.py -l 0.01 -n contrast -b ${batchsize} -o both --save_dir=${main_savedir}/test_partition

python run_script.py -l 0.02 -n contrast -b ${batchsize} -o partition --save_dir=${main_savedir}/test_partition
python run_script.py -l 0.02 -n contrast -b ${batchsize} -o patient --save_dir=${main_savedir}/test_partition
python run_script.py -l 0.02 -n contrast -b ${batchsize} -o both --save_dir=${main_savedir}/test_partition

python run_script.py -l 0.1 -n contrast -b ${batchsize} -o partition --save_dir=${main_savedir}/test_partition
python run_script.py -l 0.1 -n contrast -b ${batchsize} -o patient --save_dir=${main_savedir}/test_partition
python run_script.py -l 0.1 -n contrast -b ${batchsize} -o both --save_dir=${main_savedir}/test_partition

# test basic augmentation
python run_script.py -l 0.01 -n contrast -b ${batchsize} -a simple --save_dir=${main_savedir}/test_augment
python run_script.py -l 0.01 -n contrast -b ${batchsize} -a strong --save_dir=${main_savedir}/test_augment

python run_script.py -l 0.02 -n contrast -b ${batchsize} -a simple --save_dir=${main_savedir}/test_augment
python run_script.py -l 0.02 -n contrast -b ${batchsize} -a strong --save_dir=${main_savedir}/test_augment

python run_script.py -l 0.1 -n contrast -b ${batchsize} -a simple --save_dir=${main_savedir}/test_augment
python run_script.py -l 0.1 -n contrast -b ${batchsize} -a strong --save_dir=${main_savedir}/test_augment


# test basic group sample number
python run_script.py -l 0.02 -n contrast -b ${batchsize} -g 6 --save_dir=${main_savedir}/test_group_sample_num
python run_script.py -l 0.02 -n contrast -b ${batchsize} -g 25 --save_dir=${main_savedir}/test_group_sample_num

python run_script.py -l 0.10 -n contrast -b ${batchsize} -g 6 --save_dir=${main_savedir}/test_group_sample_num
python run_script.py -l 0.10 -n contrast -b ${batchsize} -g 25 --save_dir=${main_savedir}/test_group_sample_num