#! /bin/bash
# launch_hrl_ode_simulator.sbatch
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

##The arguments are in order of $input_dim $num_tasks $seq_length $nums_iter $lr_ws $lr_v $update_freq $logdir $seeds $identical flag

echo experiments/hrl/hrl_ode_simul.py --input-dim ${1} --num-tasks ${2} --seq-length ${3} --nums-iter ${4}\
 --lr-ws ${5} --lr-v ${6} --update-frequency ${7} --logdir ${8} --seeds ${9} ${10} 

python experiments/hrl/hrl_ode_simul.py --input-dim ${1} --num-tasks ${2} --seq-length ${3} --nums-iter ${4}\
 --lr-ws ${5} --lr-v ${6} --update-frequency ${7} --logdir ${8} --seeds ${9} ${10} 