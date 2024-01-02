#! /bin/bash

export PYTHONPATH /nfs/nhome/live/jlee/rotation_saxe/composite_learning
echo "Launch HRL perturbation experiment"

args=("$@")
input_dim=1000
nums_iter='0,1000000' 
perturbation_num_iter=200000
update_frequency=2000
w_angle=0.79
v_angle=0.79
logdir='hrl_ode_logs/220923_baseline_perturbation/'
seeds=2

set | grep '^[a-z].*='

mkdir -p $logdir

for seq_length in 8 12; do
for num_tasks in 4 8; do
for lr_ws in 1.0,1.0; do
for lr_v in 1.0 ; do
for perturbation_angle in 0.0 0.79 1.05 1.57 2.09 2.61 3.14; do
for v_norm in 0; do

sbatch experiments/hrl/launch_hrl_perturbation.sbatch ${input_dim} ${num_tasks} ${seq_length} ${v_angle} ${perturbation_num_iter} ${nums_iter} ${lr_ws} ${lr_v} ${update_frequency} ${logdir} ${seeds} ${w_angle} ${perturbation_angle} ${v_norm}

done
done
done
done
done
done