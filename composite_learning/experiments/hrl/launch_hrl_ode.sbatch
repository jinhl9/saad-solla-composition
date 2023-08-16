#!/bin/bash
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

srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/hrl/launch_hrl_ode.sh 1000 4 4 0,400000 1,1 0.1 1000 ./hrl_ode_logs/baseline_VT1/ 5 ;  sleep 1" & 
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/hrl/launch_hrl_ode.sh 1000 4 4 0,400000 1,1 0.01 1000 ./hrl_ode_logs/baseline_VT1/ 5 ;  sleep 1" & 
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/hrl/launch_hrl_ode.sh 1000 4 4 0,400000 1,1 0.001 1000 ./hrl_ode_logs/baseline_VT1/ 5 ;  sleep 1" & 
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/hrl/launch_hrl_ode.sh 1000 4 4 0,400000 1,1 0.0001 1000 ./hrl_ode_logs/baseline_VT1/ 5 ;  sleep 1" & 