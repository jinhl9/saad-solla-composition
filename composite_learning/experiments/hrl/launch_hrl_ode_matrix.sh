#! /bin/bash

export PYTHONPATH /nfs/nhome/live/jlee/rotation_saxe/composite_learning
echo "Launch HRL perturbation experiment"

args=("$@")
input_dim=1000
num_tasks=4
seq_length=4
nums_iter='400000,400000' 
update_frequency=2000
logdir='hrl_ode_logs/120923_curriculum_Q1/'
seeds=5
noise_scale=0

set | grep '^[a-z].*='

mkdir -p $logdir

for seq_length in 1 4 8; do
for lr_ws in 1.0,0.01 1.0,0.1 1.0,1.0; do
for lr_v in 1 0.1 0.01 ; do
for v_angle in 0.79 1.57 ; do
for vt_weights in 1,1,1,1 3,1,1,1 1,1,0,0 ; do

sbatch experiments/hrl/launch_hrl_ode_matrix.sbatch ${input_dim} ${num_tasks} ${seq_length} ${v_angle} ${vt_weights} ${nums_iter} ${lr_ws} ${lr_v} ${update_frequency} ${logdir} ${seeds} ${noise_scale}

done
done
done
done
done