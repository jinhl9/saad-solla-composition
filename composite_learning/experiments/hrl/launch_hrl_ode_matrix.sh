#! /bin/bash

export PYTHONPATH /nfs/nhome/live/jlee/rotation_saxe/composite_learning
echo "Launch HRL perturbation experiment"

args=("$@")
input_dim=1000
num_tasks=4
seq_length=4
nums_iter='0,400000' 
update_frequency=1000
logdir='hrl_ode_logs/baseline_160823/'
seeds=3

set | grep '^[a-z].*='

mkdir -p $logdir

for lr_ws in 0,1 0,0.1, 0,0.01; do
for lr_v in 1 0.1 0.01 0.001; do
for v_angle in 3.14 1.57 0.79; do

sbatch experiments/hrl/launch_hrl_ode_matrix.sbatch ${input_dim} ${num_tasks} ${seq_length} ${v_angle} ${nums_iter} ${lr_ws} ${lr_v} ${update_frequency} ${logdir} ${seeds}

done
done
done