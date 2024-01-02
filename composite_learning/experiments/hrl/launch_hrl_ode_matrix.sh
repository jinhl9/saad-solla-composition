#! /bin/bash

export PYTHONPATH /nfs/nhome/live/jlee/rotation_saxe/composite_learning
echo "Launch HRL perturbation experiment"

args=("$@")
input_dim=1000
nums_iter='0,500000' 
update_frequency=100
logdir='hrl_ode_logs/0612123_baseline_controlled_VS/'
seeds=3
v_norm=0
noise_scale=0

set | grep '^[a-z].*='

mkdir -p $logdir

for seq_length in 1 2 4 6 8 10; do
for num_tasks in 2; do
for lr_ws in 1.0,1.0; do
for lr_v in 1.0 ; do
for vs in 0.9,0.1 0.8,0.2 0.7,0.3 0.5,0.5; do
for vt_weights in 1,1,1,1 ; do

sbatch experiments/hrl/launch_hrl_ode_matrix.sbatch ${input_dim} ${num_tasks} ${seq_length} ${vs} ${vt_weights} ${nums_iter} ${lr_ws} ${lr_v} ${update_frequency} ${logdir} ${seeds} ${noise_scale} ${v_norm}

done
done
done
done
done
done