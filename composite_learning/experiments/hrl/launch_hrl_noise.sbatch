#! /bin/bash
# launch_hrl_noise.sbatch
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

echo experiments/hrl/hrl_simpler.py --input-dim ${1} --num-tasks ${2} --seq-length ${3} --vs ${4} --vt ${5} --nums-iter ${6}\
 --lr-ws ${7} --lr-v ${8} --update-frequency ${9} --logdir ${10} --seeds ${11} --noise-scale ${12} --v-norm ${13} --w-noise ${14} --v-noise ${15} --ode 0 --sim 1

python experiments/hrl/hrl_simpler.py --input-dim ${1} --num-tasks ${2} --seq-length ${3} --vs ${4} --vt ${5} --nums-iter ${6}\
 --lr-ws ${7} --lr-v ${8} --update-frequency ${9} --logdir ${10} --seeds ${11} --noise-scale ${12} --v-norm ${13} --w-noise ${14} --v-noise ${15} --ode 0 --sim 1
