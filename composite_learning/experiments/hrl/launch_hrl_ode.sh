#! /bin/bash

export PYTHONPATH '/nfs/nhome/live/jlee/rotation_saxe/composite_learning'
echo "Launch HRL ODEs"
args=("$@")
input_dim=${args[0]}
num_tasks=${args[1]}
seq_length=${args[2]}
nums_iter=${args[3]}
lr_ws=${args[4]}
lr_v=${args[5]}
update_freq=${args[6]}
logdir=${args[7]}
seeds=${args[8]}
identical=${args[9]}

set | grep '^[a-z].*='

mkdir -p $logdir

python experiments/hrl/hrl_ode.py --input-dim ${input_dim} --num-tasks ${num_tasks} --seq-length ${seq_length} --nums-iter ${nums_iter}\
 --lr-ws ${lr_ws} --lr-v ${lr_v} --update-frequency ${update_freq} --logdir ${logdir} --seeds ${seeds} ${identical}
