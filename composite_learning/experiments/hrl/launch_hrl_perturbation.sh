#! /bin/bash

export PYTHONPATH /nfs/nhome/live/jlee/rotation_saxe/composite_learning
echo "Launch HRL perturbation experiment"

args=("$@")
rootpath=${args[0]}
perturbation_level=${args[1]}
num_iter=${args[2]}
lr_w=${args[3]}
lr_v=${args[4]}
update_frequency=${args[5]}
seeds=${args[6]}

set | grep '^[a-z].*='

for logpath in "$rootpath"*/simulator_*.jl ; do
if ! [[ $logpath == *"weights"* ]] ; then
    sbatch experiments/hrl/launch_hrl_perturbation.sbatch ${logpath} ${perturbation_level} ${num_iter} ${lr_w} ${lr_v} ${update_frequency} ${seeds} 
fi
done
