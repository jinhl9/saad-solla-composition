#!/bin/bash
#
#SBATCH --job-name=ChainRL
#SBATCH -p cpu
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -t 0-3:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G   
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#

srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_chain_rl.sh 3 200000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_chain_rl.sh 6 200000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_chain_rl.sh 5 200000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_chain_rl.sh 7 500000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_chain_rl.sh 8 500000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_chain_rl.sh 9 500000" &
