#! /bin/bash

export PYTHONPATH /nfs/nhome/live/jlee/rotation_saxe/composite_learning
echo "Launch HRL perturbation experiment"

args=("$@")
input_dim=1000
max_iters='5000000,1000000'
v_norm=0
w_angle=1.55
logdir='hrl_ode_logs/020124_curriculum_training_time_controlled_vs/'
seeds=2

set | grep '^[a-z].*='

mkdir -p $logdir

for seq_length in 1 2 4 6 8 10; do
for num_tasks in 4 ; do
for lr_w1 in 10. 2. 1. 0.5 0.1 0.01; do 
for lr_w2 in 10. 2. 1. 0.5 0.1 0.01; do
for lr_v in 10. 2. 1. 0.5 0.1 0.01; do 
for vs in 0.8,0.1,0.05,0.05 0.4,0.3,0.2,0.1 0.7,0.1,0.1,0.1 ; do 
#for vs in 0.99,0.01 0.9,0.1 0.75,0.25 0.6,0.4 ; do 
sbatch experiments/hrl/launch_lr_search_training_time.sbatch ${input_dim} ${num_tasks} ${seq_length} ${vs} ${w_angle} ${max_iters} ${lr_w1} ${lr_w2} ${lr_v} ${logdir} ${seeds}

done
done
done
done
done
done