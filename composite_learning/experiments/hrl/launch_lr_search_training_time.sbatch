#! /bin/bash
# launch_hrl_training_time.sbatch
#
#SBATCH --job-name=jlee/HRL_ODE
#SBATCH -p cpu
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -t 0-8:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G   
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

##The arguments are in order of ${input_dim} ${num_tasks} ${seq_length} ${v_angle} ${max_iters} ${lr_ws} ${lr_v} ${logdir} ${seeds}

echo experiments/hrl/training_time.py --input-dim ${1} --num-tasks ${2} --seq-length ${3} --vs ${4} --vt ${5} --w-angle ${6} --max-iters ${7}\
 --lr-w1 ${8} --lr-w2 ${9} --lr-v ${10} --logdir ${11} --seeds ${12} --simulation ${13} --ode ${14}

python experiments/hrl/training_time.py --input-dim ${1} --num-tasks ${2} --seq-length ${3} --vs ${4} --vt ${5} --w-angle ${6} --max-iters ${7}\
 --lr-w1 ${8} --lr-w2 ${9} --lr-v ${10} --logdir ${11} --seeds ${12} --simulation ${13} --ode ${14}