echo "Launch HRL perturbation experiment"

args=("$@")
input_dim=1000
nums_iter='0,1000000' 
update_frequency=100
logdir='hrl_ode_logs/190124_vanilla_figure/'
seeds=10
v_norm=0
noise_scale=0
vt='0.4,0.3,0.2,0.1'
w_noise=0
v_noise=0

set | grep '^[a-z].*='

mkdir -p $logdir

for seq_length in 4 6 8 10; do
for num_tasks in 4; do
for lr_ws in 1.0,1.0; do
for lr_v in 1.0 ; do
for vs in 0.06353486232558558,0.09349115002497735,0.825564922031083,0.5528700440863908; do
#0.7302967433402215,0.5477225575051662,0.36514837167011077,0.18257418583505539 0.9017228224031859,0.007659691385011103,0.023212892437916355,0.4316230326451366 0.03508969335751355,0.21312709370117328,0.9763749620726612,0.006122808846934363 0.012456880909664347,0.04523171968488168,0.09469158887965982,0.9944005333102938; do
#for vs in 0.9788,-0.2045 0.8880,-0.4597 0.6951,-0.7189 0.4773,-0.8787 0.3383,-0.9410; do

sbatch experiments/hrl/launch_hrl_noise.sbatch ${input_dim} ${num_tasks} ${seq_length} ${vs} ${vt} ${nums_iter} ${lr_ws} ${lr_v} ${update_frequency} ${logdir} ${seeds} ${noise_scale} ${v_norm} ${w_noise} ${v_noise}

done
done
done
done
done