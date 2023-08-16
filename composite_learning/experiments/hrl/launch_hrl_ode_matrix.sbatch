#! /bin/bash
# launch_hrl_ode_matrix.sbatch
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

echo experiments/hrl/hrl_simpler.py --input-dim ${1} --num-tasks ${2} --seq-length ${3} --v-angle ${4} --nums-iter ${5}\
 --lr-ws ${6} --lr-v ${7} --update-frequency ${8} --logdir ${9} --seeds ${10}

python experiments/hrl/hrl_simpler.py --input-dim ${1} --num-tasks ${2} --seq-length ${3} --v-angle ${4} --nums-iter ${5}\
 --lr-ws ${6} --lr-v ${7} --update-frequency ${8} --logdir ${9} --seeds ${10}