#!/usr/bin/env bash
# test if using different subhead vs num_clusters num works


# change subheads
python -O  run_script_iic.py -l 0.01 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=1 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=1 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=1 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=1 --time 4 --save_dir=0805/test_subheads_vs_num_clusters

python -O  run_script_iic.py -l 0.01 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=10 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=10 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=10 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=10 --time 4 --save_dir=0805/test_subheads_vs_num_clusters

python -O  run_script_iic.py -l 0.01 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=20 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=20 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=20 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 10 -t 1.0 -g 25 --num_subheads=20 --time 4 --save_dir=0805/test_subheads_vs_num_clusters

# change num_clusters to 2
python -O  run_script_iic.py -l 0.01 -n iiccontrast -b 200 -o partition -c 20 -t 1.0 -g 25 --num_subheads=1 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 20 -t 1.0 -g 25 --num_subheads=1 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 20 -t 1.0 -g 25 --num_subheads=1 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 20 -t 1.0 -g 25 --num_subheads=1 --time 4 --save_dir=0805/test_subheads_vs_num_clusters

python -O  run_script_iic.py -l 0.01 -n iiccontrast -b 200 -o partition -c 20 -t 1.0 -g 25 --num_subheads=10 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 20 -t 1.0 -g 25 --num_subheads=10 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 20 -t 1.0 -g 25 --num_subheads=10 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 20 -t 1.0 -g 25 --num_subheads=10 --time 4 --save_dir=0805/test_subheads_vs_num_clusters

python -O  run_script_iic.py -l 0.01 -n iiccontrast -b 200 -o partition -c 20 -t 1.0 -g 25 --num_subheads=20 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 20 -t 1.0 -g 25 --num_subheads=20 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 20 -t 1.0 -g 25 --num_subheads=20 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 20 -t 1.0 -g 25 --num_subheads=20 --time 4 --save_dir=0805/test_subheads_vs_num_clusters


# change num_clusters to 50
python -O  run_script_iic.py -l 0.01 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=1 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=1 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=1 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=1 --time 4 --save_dir=0805/test_subheads_vs_num_clusters

python -O  run_script_iic.py -l 0.01 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=10 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=10 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=10 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=10 --time 4 --save_dir=0805/test_subheads_vs_num_clusters

python -O  run_script_iic.py -l 0.01 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=20 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.02 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=20 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.05 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=20 --time 4 --save_dir=0805/test_subheads_vs_num_clusters
python -O  run_script_iic.py -l 0.10 -n iiccontrast -b 200 -o partition -c 50 -t 1.0 -g 25 --num_subheads=20 --time 4 --save_dir=0805/test_subheads_vs_num_clusters

