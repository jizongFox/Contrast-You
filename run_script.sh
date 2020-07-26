#!/usr/bin/env bash

python run_script.py -l 0.02 -n contrast -b 500 -o partition
#python run_script.py -l 0.02 -n contrast -b 500 -o patient
#python run_script.py -l 0.02 -n contrast -b 500 -o both

python run_script.py -l 0.02 -n contrastMT -b 500 -o partition -w 10
#python run_script.py -l 0.02 -n contrastMT -b 500 -o patient -w 10
#python run_script.py -l 0.02 -n contrastMT -b 500 -o both -w 10


python run_script.py -l 0.05 -n contrast -b 500 -o partition
#python run_script.py -l 0.05 -n contrast -b 500 -o patient
#python run_script.py -l 0.05 -n contrast -b 500 -o both

python run_script.py -l 0.05 -n contrastMT -b 500 -o partition -w 10
#python run_script.py -l 0.05 -n contrastMT -b 500 -o patient -w 10
#python run_script.py -l 0.05 -n contrastMT -b 500 -o both -w 10

python run_script.py -l 0.1 -n contrast -b 500 -o partition
#python run_script.py -l 0.1 -n contrast -b 500 -o patient
#python run_script.py -l 0.1 -n contrast -b 500 -o both

python run_script.py -l 0.1 -n contrastMT -b 500 -o partition -w 10
#python run_script.py -l 0.1 -n contrastMT -b 500 -o patient -w 10
#python run_script.py -l 0.1 -n contrastMT -b 500 -o both -w 10



python run_script.py -l 1.0 -n contrast -b 500 -o partition
#python run_script.py -l 1.0 -n contrast -b 500 -o patient
#python run_script.py -l 1.0 -n contrast -b 500 -o both

python run_script.py -l 1.0 -n contrastMT -b 500 -o partition -w 10
#python run_script.py -l 1.0 -n contrastMT -b 500 -o patient -w 10
#python run_script.py -l 1.0 -n contrastMT -b 500 -o both -w 10
