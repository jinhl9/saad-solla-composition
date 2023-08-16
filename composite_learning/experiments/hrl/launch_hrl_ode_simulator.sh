#! /bin/bash

export PYTHONPATH '/nfs/nhome/live/jlee/rotation_saxe/composite_learning'
echo "Launch HRL ODEs"
args=("$@")
input_dim=1000
num_tasks=4
seq_length=4
nums_iter='200000,400000' 
update_frequency=1000
logdir='hrl_ode_logs/curriculum_VT1_20230810/'
seeds=5

set | grep '^[a-z].*='

mkdir -p $logdir

for lr_ws in 10,1 10,0.1 10,0.01; do
for lr_v in 1 0.1 0.01 0.001; do
for identical in '--identical' ''; do
echo ${lr_ws}
sbatch experiments/hrl/launch_hrl_ode_simulator.sbatch ${input_dim} ${num_tasks} ${seq_length} ${nums_iter} ${lr_ws} ${lr_v} ${update_frequency} ${logdir} ${seeds} ${identical} 

done
done
done