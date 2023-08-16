#! /bin/bash
# launch_hrl_perturbation.sbatch
#
#SBATCH --job-name=jlee/HRL_ODE
#SBATCH -p cpu
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -t 0-8:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH -e slurm.%N.%j.err # STDERR  

echo python experiments/hrl/hrl_perturbation.py --logpath ${1} --perturbation-level ${2} --num-iter ${3} --lr-w ${4} --lr-v ${5} --update-frequency ${6} --seeds ${7} 

python experiments/hrl/hrl_perturbation.py --logpath ${1} --perturbation-level ${2} --num-iter ${3} --lr-w ${4} --lr-v ${5} --update-frequency ${6} --seeds ${7} 