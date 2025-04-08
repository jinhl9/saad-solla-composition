#! /bin/bash

export PYTHONPATH /nfs/nhome/live/jlee/rotation_saxe/composite_learning
echo "Launch HRL perturbation experiment"

args=("$@")
input_dim=1000
max_iters='0,800000'
v_norm=0
logdir='hrl_ode_logs/20012024_noisy_baseline_training_time/'
seeds=3
lr_w1=1
lr_w2=1
lr_v=1
simulation=1
ode=0
vt='0.8,0.2'
set | grep '^[a-z].*='

mkdir -p $logdir

for seq_length in 6 8 10; do
for num_tasks in 2 ; do
for w_noise in 0.05 0.01 0.005; do 
for v_noise in 0.05 0.01 0.005; do 
for vs in 0.7674,0.6411 0.5671,0.8235 0.2750,0.9614 0.0077,0.99997; do
for w_angle in 1.57; do
python experiments/hrl/training_time.py --input-dim ${input_dim} --num-tasks ${num_tasks} --seq-length ${seq_length} --vs ${vs} --vt ${vt} --w-angle ${w_angle} --max-iters ${max_iters}\
 --lr-w1 ${lrw_w1} --lr-w2 ${lr_w2} --lr-v ${lr_v} --logdir ${logdir} --seeds ${seeds} --simulation ${simulation} --ode ${ode} --w-noise ${w_noise} --v-noise ${v_noise}

#sbatch experiments/hrl/launch_lr_search_training_time.sbatch ${input_dim} ${num_tasks} ${seq_length} ${vs} ${vt} ${w_angle} ${max_iters} ${lr_w1} ${lr_w2} ${lr_v} ${logdir} ${seeds} ${simulation} ${ode} ${w_noise} ${v_noise}

done
done
done
done
done
done