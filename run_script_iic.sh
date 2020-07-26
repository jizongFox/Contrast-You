#!/usr/bin/env bash
python run_script_iic.py -l 0.02 -n iiccontrast -b 500 -o partition
python run_script_iic.py -l 0.05 -n iiccontrast -b 500 -o partition
python run_script_iic.py -l 0.1 -n iiccontrast -b 500 -o partition
python run_script_iic.py -l 1.0 -n iiccontrast -b 500 -o partition




