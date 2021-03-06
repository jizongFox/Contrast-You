#!/bin/bash
set -e pipefail
batch_size=2
epoch=2
rand_seed=10
save_dir=test_pipeline
group_num=3
comm_cmd=" --on-local"

python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=2 \
 baseline


## infonce only global
#python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=3 \
#   infonce --global_features Conv5 --global_importance 1
#
##### infonce
## encoder + decoder
#python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=3 \
#   infonce --global_features Conv5 --global_importance 1 --dense_features Conv5 --dense_importance 1 -g=${group_num}
#
## encoder + decoder
#python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=3 \
#   infonce --global_features Conv5 --global_importance 1 --dense_features Conv5 --dense_importance 0.1 -g=${group_num}
#
## encoder + decoder
#python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=3 \
#   infonce --global_features Conv5 --global_importance 1 --dense_features Conv5 --dense_importance 0.01 -g=${group_num}
#
## encoder + decoder
#python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=3 \
#   infonce --global_features Conv5 --global_importance 1 --dense_features Conv5 --dense_importance 0.001 -g=${group_num}
#
##### soften infonce
## encoder + decoder
#python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=3 \
#   softeninfonce --global_features Conv5 --global_importance 1 --dense_features Conv5 --dense_importance 1 \
#  --softenweight 1 0.1 0.01 0.001 0.0001 -g=${group_num}
#
## encoder + decoder
#python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=3 \
#   softeninfonce --global_features Conv5 --global_importance 1 --dense_features Conv5 --dense_importance 0.1 \
#  --softenweight 1 0.1 0.01 0.001 0.0001 -g=${group_num}
#
## encoder + decoder
#python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=3 \
#   softeninfonce --global_features Conv5 --global_importance 1 --dense_features Conv5 --dense_importance 0.01 \
#  --softenweight 1 0.1 0.01 0.001 0.0001 -g=${group_num}
#
## encoder + decoder
#python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=3 \
#   softeninfonce --global_features Conv5 --global_importance 1 --dense_features Conv5 --dense_importance 0.001 \
#  --softenweight 1 0.1 0.01 0.001 0.0001 -g=${group_num}
#
#### mixup
#python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=4 \
#   mixup --global_features Conv5 --global_importance 1 --softenweight 1 0.1 0.01 0.001 0.0001 -g=${group_num}

#### multitask
#python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=4 \
#   multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.5 0.1 --contrast_on partition patient cycle -g=${group_num}
#
#python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=4 \
#   multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.1 0.05 --contrast_on partition patient cycle -g=${group_num}
#
#python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=4 \
#   multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.01 0.01 --contrast_on partition patient cycle -g=${group_num}


python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=4 \
 multitask --global_features Conv5  --global_importance 1  --contrast_on partition  -g=${group_num}

python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=4 \
 multitask --global_features Conv5  --global_importance 1  --contrast_on patient  -g=${group_num}

 python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=4 \
 multitask --global_features Conv5  --global_importance 1  --contrast_on cycle  -g=${group_num}

 python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${batch_size} -e ${epoch} -s ${rand_seed} --time=4 \
   multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.1 0.1 --contrast_on partition patient cycle -g=${group_num}