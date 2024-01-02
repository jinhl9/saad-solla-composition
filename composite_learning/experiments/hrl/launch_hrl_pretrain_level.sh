#! /bin/bash

export PYTHONPATH /nfs/nhome/live/jlee/rotation_saxe/composite_learning
echo "Launch HRL perturbation experiment"

args=("$@")
input_dim=1000
num_tasks=4
seq_length=4
nums_iter='0,1000000' 
update_frequency=2000
logdir='hrl_ode_logs/220923_curriculum_pretrainlevel/'
seeds=2
vt_weights='1,1,1,1'

set | grep '^[a-z].*='

mkdir -p $logdir

for seq_length in 8 12; do
for num_tasks in 4 8; do
for lr_ws in 1.0,1.0; do
for lr_v in 1.0 ; do
for v_angle in 0.0 0.79 1.05 1.57 2.09 2.61 3.14 ; do
for w_angle in 0.0 0.79 1.05 1.57 2.09 2.61 3.14 ; do
for v_norm in 0; do

sbatch experiments/hrl/launch_hrl_pretrain_level.sbatch ${input_dim} ${num_tasks} ${seq_length} ${v_angle} ${vt_weights} ${nums_iter} ${lr_ws} ${lr_v} ${update_frequency} ${logdir} ${seeds} ${w_angle} ${v_norm}

done
done
done
done
done
done
done