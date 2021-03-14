#!/bin/bash
set -e pipefail
rand_seed=10
save_dir=0312
group_num=10
num_batches=200
pre_max_epoch=120
ft_max_epoch=80
comm_cmd=" --on-local "

python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=2 \
  baseline -e ${ft_max_epoch}

#### multitask
python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=4 \
  multitask --global_features Conv5 --global_importance 1 --contrast_on partition -g=${group_num} \
  -pe ${pre_max_epoch} -fe ${ft_max_epoch}

python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=4 \
  multitask --global_features Conv5 --global_importance 1 --contrast_on patient -g=${group_num} \
  -pe ${pre_max_epoch} -fe ${ft_max_epoch}

python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=4 \
  multitask --global_features Conv5 --global_importance 1 --contrast_on cycle -g=${group_num} \
  -pe ${pre_max_epoch} -fe ${ft_max_epoch}

## multitask
python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=4 \
  multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.5 0.1 \
  --contrast_on partition patient cycle -g=${group_num} -pe ${pre_max_epoch} -fe ${ft_max_epoch}

python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=4 \
  multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.1 0.05 \
  --contrast_on partition patient cycle -g=${group_num} -pe ${pre_max_epoch} -fe ${ft_max_epoch}

python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=4 \
  multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.01 0.01 \
  --contrast_on partition patient cycle -g=${group_num} -pe ${pre_max_epoch} -fe ${ft_max_epoch}

python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=4 \
  multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.1 0.03 \
  --contrast_on partition patient cycle -g=${group_num} -pe ${pre_max_epoch} -fe ${ft_max_epoch}

python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=4 \
  multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.1 0.1 \
  --contrast_on partition patient cycle -g=${group_num} -pe ${pre_max_epoch} -fe ${ft_max_epoch}

## multitask with dense
python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=4 \
  multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.5 0.1 \
  --dense_features Conv5 --dense_importance 0.05 \
  --contrast_on partition patient cycle -g=${group_num} -pe ${pre_max_epoch} -fe ${ft_max_epoch}

python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=4 \
  multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.1 0.05 \
  --dense_features Conv5 --dense_importance 0.05 \
  --contrast_on partition patient cycle -g=${group_num} -pe ${pre_max_epoch} -fe ${ft_max_epoch}

python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=4 \
  multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.01 0.01 \
  --dense_features Conv5 --dense_importance 0.05 \
  --contrast_on partition patient cycle -g=${group_num} -pe ${pre_max_epoch} -fe ${ft_max_epoch}

python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=4 \
  multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.1 0.03 \
  --dense_features Conv5 --dense_importance 0.05 \
  --contrast_on partition patient cycle -g=${group_num} -pe ${pre_max_epoch} -fe ${ft_max_epoch}

python run_script_0228.py ${comm_cmd} --save_dir ${save_dir} -n acdc -b ${num_batches} -s ${rand_seed} --time=4 \
  multitask --global_features Conv5 Conv5 Conv5 --global_importance 1 0.1 0.1 \
  --dense_features Conv5 --dense_importance 0.05 \
  --contrast_on partition patient cycle -g=${group_num} -pe ${pre_max_epoch} -fe ${ft_max_epoch}
