#!/usr/bin/env bash
# this script check the shuffule option on contrastive sampler
#python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 10 -t 1.0 -g 6  --save_dir=check_shuffle_no
#python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 10 -t 1.0 -g 6 --contrast_shuffle --save_dir=check_shuffle_yes

# this script check the shuffle option in contrastivesampler, large batch size and large num_clusters

python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 10 -t 1.0 -g 6
python run_script_iic.py -l 0.05 -n iiccontrast -b 500 -o partition -c 10 -t 1.0 -g 6
python run_script_iic.py -l 0.1 -n iiccontrast -b 500 -o partition -c 10 -t 1.0 -g 6

python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 50 -t 1.0 -g 6
python run_script_iic.py -l 0.05 -n iiccontrast -b 500 -o partition -c 50 -t 1.0 -g 6
python run_script_iic.py -l 0.1 -n iiccontrast -b 500 -o partition -c 50 -t 1.0 -g 6

python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 80 -t 1.0 -g 6
python run_script_iic.py -l 0.05 -n iiccontrast -b 500 -o partition -c 80 -t 1.0 -g 6
python run_script_iic.py -l 0.1 -n iiccontrast -b 500 -o partition -c 80 -t 1.0 -g 6


python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 10 -t 1.0 -g 25 --time 8
python run_script_iic.py -l 0.05 -n iiccontrast -b 500 -o partition -c 10 -t 1.0 -g 25 --time 8
python run_script_iic.py -l 0.1 -n iiccontrast -b 500 -o partition -c 10 -t 1.0 -g 25 --time 8

python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 50 -t 1.0 -g 25 --time 8
python run_script_iic.py -l 0.05 -n iiccontrast -b 500 -o partition -c 50 -t 1.0 -g 25 --time 8
python run_script_iic.py -l 0.1 -n iiccontrast -b 500 -o partition -c 50 -t 1.0 -g 25 --time 8

python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 80 -t 1.0 -g 25 --time 8
python run_script_iic.py -l 0.05 -n iiccontrast -b 500 -o partition -c 80 -t 1.0 -g 25 --time 8
python run_script_iic.py -l 0.1 -n iiccontrast -b 500 -o partition -c 80 -t 1.0 -g 25 --time 8
