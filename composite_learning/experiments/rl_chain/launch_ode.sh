#! /bin/bash

echo "Launch chained RL vs. baseline ode simulation with multiple chains"
args=("$@")
num_chain=${args[0]}
seq_len=${args[1]}
chain_step=${args[2]}
train_step=${args[3]}
update=${args[4]}


echo "num_chain: ${num_chain}, seq_len: ${seq_len}, step: ${train_step}"
python experiments/rl_chain/run_rl_ode.py --num-chain ${num_chain} --update-frequency ${update} --seq-length ${seq_len}\
 --logdir ode_logs_final/len_${seq_len}_chain_${num_chain}/ --train-per-chain ${chain_step} --total-iter ${train_step} --chained --baseline

