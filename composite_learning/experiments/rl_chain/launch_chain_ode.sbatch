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

srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 2 11 5000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 2 12 5000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 2 13 5000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 2 14 5000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 2 15 5000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 2 16 5000000 10000000 10000" &

srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 2 5 5000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 2 6 5000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 2 7 5000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 2 8 5000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 2 9 5000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 2 10 5000000 10000000 10000" &

srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 3 3 2000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 4 3 2000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 5 3 2000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 6 3 2000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 7 3 2000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 8 3 2000000 10000000 10000" &
srun --ntasks=1 --nodes=1 --cpus-per-task=1 bash -c "./experiments/rl_chain/launch_ode.sh 9 3 2000000 10000000 10000" &
