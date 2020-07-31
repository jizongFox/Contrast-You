#!/usr/bin/env bash
set -e
# test basic partition
python run_script.py -l 0.01 -n contrast -b 500 -o partition --save_dir=test_partition
python run_script.py -l 0.01 -n contrast -b 500 -o patient --save_dir=test_partition
python run_script.py -l 0.01 -n contrast -b 500 -o both --save_dir=test_partition

python run_script.py -l 0.02 -n contrast -b 500 -o partition --save_dir=test_partition
python run_script.py -l 0.02 -n contrast -b 500 -o patient --save_dir=test_partition
python run_script.py -l 0.02 -n contrast -b 500 -o both --save_dir=test_partition

python run_script.py -l 0.1 -n contrast -b 500 -o partition --save_dir=test_partition
python run_script.py -l 0.1 -n contrast -b 500 -o patient --save_dir=test_partition
python run_script.py -l 0.1 -n contrast -b 500 -o both --save_dir=test_partition

# test basic augmentation
python run_script.py -l 0.02 -n contrast -b 500 -a simple --save_dir=test_augment
python run_script.py -l 0.02 -n contrast -b 500 -a strong --save_dir=test_augment

# test basic group sample number
python run_script.py -l 0.02 -n contrast -b 500 -g 6 --save_dir=test_group_sample_num
python run_script.py -l 0.02 -n contrast -b 500 -g 25 --save_dir=test_group_sample_num

#python run_script.py -l 0.02 -n contrastMT -b 500 -o partition -w 10
#python run_script.py -l 0.02 -n contrastMT -b 500 -o patient -w 10
#python run_script.py -l 0.02 -n contrastMT -b 500 -o both -w 10
#
#
#python run_script.py -l 0.05 -n contrast -b 500 -o partition
#python run_script.py -l 0.05 -n contrast -b 500 -o patient
#python run_script.py -l 0.05 -n contrast -b 500 -o both
#
#python run_script.py -l 0.05 -n contrastMT -b 500 -o partition -w 10
#python run_script.py -l 0.05 -n contrastMT -b 500 -o patient -w 10
#python run_script.py -l 0.05 -n contrastMT -b 500 -o both -w 10
#
#python run_script.py -l 0.1 -n contrast -b 500 -o partition
#python run_script.py -l 0.1 -n contrast -b 500 -o patient
#python run_script.py -l 0.1 -n contrast -b 500 -o both
#
#python run_script.py -l 0.1 -n contrastMT -b 500 -o partition -w 10
#python run_script.py -l 0.1 -n contrastMT -b 500 -o patient -w 10
#python run_script.py -l 0.1 -n contrastMT -b 500 -o both -w 10

#python run_script.py -l 1.0 -n contrast -b 500 -o partition
#python run_script.py -l 1.0 -n contrast -b 500 -o patient
#python run_script.py -l 1.0 -n contrast -b 500 -o both

#python run_script.py -l 1.0 -n contrastMT -b 500 -o partition -w 10
#python run_script.py -l 1.0 -n contrastMT -b 500 -o patient -w 10
#python run_script.py -l 1.0 -n contrastMT -b 500 -o both -w 10
