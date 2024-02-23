#! /bin/bash

export PYTHONPATH /nfs/nhome/live/jlee/rotation_saxe/composite_learning
echo "Launch HRL perturbation experiment"

args=("$@")
input_dim=1000
nums_iter='0,500000' 
update_frequency=100
logdir='hrl_ode_logs/270124_noise_vanilla/'
seeds=2
v_norm=0
noise_scale=0
vt='0.8,0.2'

set | grep '^[a-z].*='

mkdir -p $logdir

for seq_length in 6; do
for num_tasks in 2; do
for lr_ws in 1.0,1.0; do
for lr_v in 1.0 ; do
for vs in 0.5671,0.8235 0.0077,0.99997 0.27502924,0.96143586; do
for w_noise in 0.2 0.3; do 
for v_noise in 0.1 0.05 0.01 0.005 0.001; do 

sbatch experiments/hrl/launch_hrl_noise.sbatch ${input_dim} ${num_tasks} ${seq_length} ${vs} ${vt} ${nums_iter} ${lr_ws} ${lr_v} ${update_frequency} ${logdir} ${seeds} ${noise_scale} ${v_norm} ${w_noise} ${v_noise}

done
done
done
done
done
done
done