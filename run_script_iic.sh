#!/usr/bin/env bash
python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 10 -t 1.0
python run_script_iic.py -l 0.05 -n iiccontrast -b 500 -o partition -c 10 -t 1.0
python run_script_iic.py -l 0.1 -n iiccontrast -b 500 -o partition -c 10 -t 1.0

python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 20 -t 1.0
python run_script_iic.py -l 0.05 -n iiccontrast -b 500 -o partition -c 20 -t 1.0
python run_script_iic.py -l 0.1 -n iiccontrast -b 500 -o partition -c 20 -t 1.0


python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 10 -t 10.0
python run_script_iic.py -l 0.05 -n iiccontrast -b 500 -o partition -c 10 -t 10.0
python run_script_iic.py -l 0.1 -n iiccontrast -b 500 -o partition -c 10 -t 10.0

python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 20 -t 10.0
python run_script_iic.py -l 0.05 -n iiccontrast -b 500 -o partition -c 20 -t 10.0
python run_script_iic.py -l 0.1 -n iiccontrast -b 500 -o partition -c 20 -t 10.0


python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 10 -t 0.1
python run_script_iic.py -l 0.05 -n iiccontrast -b 500 -o partition -c 10 -t 0.1
python run_script_iic.py -l 0.1 -n iiccontrast -b 500 -o partition -c 10 -t 0.1

python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition -c 20 -t 0.1
python run_script_iic.py -l 0.05 -n iiccontrast -b 500 -o partition -c 20 -t 0.1
python run_script_iic.py -l 0.1 -n iiccontrast -b 500 -o partition -c 20 -t 0.1



