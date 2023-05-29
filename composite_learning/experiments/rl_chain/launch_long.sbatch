#!/bin/bash
#
#SBATCH --job-name=ChainRL
#SBATCH -p cpu
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -t 0-5:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G   
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

#srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_long_chain_rl.sh 2 11 1000000 10000000 0.99 10000 --baseline --chained" &
#srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_long_chain_rl.sh 2 13 1000000 10000000 0.99 10000 --baseline --chained" &
#srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_long_chain_rl.sh 2 15 1000000 10000000 0.99 10000 --baseline --chained" &
#srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_long_chain_rl.sh 2 17 1000000 10000000 0.99 10000 --baseline --chained" &
#srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_long_chain_rl.sh 2 19 1000000 10000000 0.99 10000 --baseline --chained" &


srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_long_chain_rl.sh 5 3 500000 500000 0.99 10000 --baseline" &
#srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_long_chain_rl.sh 11 3 500000 10000000 0.99 10000 --baseline --chained" &
#srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_long_chain_rl.sh 13 3 500000 10000000 0.99 10000 --baseline --chained" &
#srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_long_chain_rl.sh 9 3 500000 10000000 0.99 10000 --baseline --chained" &
#srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_long_chain_rl.sh 15 3 500000 10000000 0.99 10000 --baseline --chained" &
#srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_long_chain_rl.sh 17 3 500000 10000000 0.99 10000 --baseline --chained" &

