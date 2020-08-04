#!/usr/bin/env bash
# test if using different subhead vs num_clusters num works


# change subheads
python run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=1 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=1 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=1 --time 6 --save_dir=0804/test_subheads_vs_num_clusters

python run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=10 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=10 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=10 --time 6 --save_dir=0804/test_subheads_vs_num_clusters

python run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=20 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=20 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=20 --time 6 --save_dir=0804/test_subheads_vs_num_clusters

# change num_clusters to 2
python run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 2 -t 1.0 -g 25 --num_subheads=1 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 2 -t 1.0 -g 25 --num_subheads=1 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 2 -t 1.0 -g 25 --num_subheads=1 --time 6 --save_dir=0804/test_subheads_vs_num_clusters

python run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 2 -t 1.0 -g 25 --num_subheads=10 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 2 -t 1.0 -g 25 --num_subheads=10 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 2 -t 1.0 -g 25 --num_subheads=10 --time 6 --save_dir=0804/test_subheads_vs_num_clusters

python run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 2 -t 1.0 -g 25 --num_subheads=20 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 2 -t 1.0 -g 25 --num_subheads=20 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 2 -t 1.0 -g 25 --num_subheads=20 --time 6 --save_dir=0804/test_subheads_vs_num_clusters


# change num_clusters to 50
python run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=1 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=1 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=1 --time 6 --save_dir=0804/test_subheads_vs_num_clusters

python run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=10 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=10 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=10 --time 6 --save_dir=0804/test_subheads_vs_num_clusters

python run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=20 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=20 --time 6 --save_dir=0804/test_subheads_vs_num_clusters
python run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=20 --time 6 --save_dir=0804/test_subheads_vs_num_clusters

