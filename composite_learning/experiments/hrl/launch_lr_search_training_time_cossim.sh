#! /bin/bash

export PYTHONPATH /nfs/nhome/live/jlee/rotation_saxe/composite_learning
echo "Launch HRL perturbation experiment"

args=("$@")
input_dim=1000
max_iters='0,10000000'
v_norm=0
logdir='hrl_ode_logs/260124_baseline_k4/'
seeds=5
vt='0.4,0.3,0.2,0.1'
simulation=0
ode=1

set | grep '^[a-z].*='

mkdir -p $logdir

for seq_length in 4 6 8 10; do
for num_tasks in 4 ; do
for lr_w1 in 1.; do #10. 1. 0.1 0.01 ; do 
for lr_w2 in 1.; do #10. 1. 0.1 0.01 ; do
for lr_v in 1.; do #10. 1. 0.1 0.01 ; do 
#for vs in 0.95,0.05,0.03,0.02 0.7,0.1,0.1,0.1 ; do #for K=4
#for vs in 0.93755,0.34027,0.05726,0.04393 0.99213,-0.01101,0.02516,0.12210 0.94416,-0.15527,-0.29046,0.00874 0.81303,-0.56156,-0.13540,0.07269 0.75031,-0.49857,-0.35156,-0.25466; do # 0.9,0.1 0.75,0.25 0.6,0.4 ; do # for K=2
for cossim in 0.8 0.6 0.4 0.2 ; do 
#0.09057639077463218,0.9958895106557965; do
for w_angle in 1.57; do
sbatch experiments/hrl/launch_lr_search_training_time_cossim.sbatch ${input_dim} ${num_tasks} ${seq_length} ${cossim} ${vt} ${w_angle} ${max_iters} ${lr_w1} ${lr_w2} ${lr_v} ${logdir} ${seeds} ${simulation} ${ode}

done
done
done
done
done
done
done