#! /bin/bash
echo "Launch chained RL vs. baseline exp with multiple chains"
args=("$@")
num_chain=${args[0]}
seq_len=${args[1]}
chain_train_step=${args[2]}
baseline_train_step=${args[3]}
threshold=${args[4]}
update=${args[5]}
baseline=${args[6]}
chained=${args[7]}

echo "num_chain: ${num_chain}, seq_len: ${seq_len}, step: ${train_step}"
python experiments/rl_chain/run_rl.py --num-chain ${num_chain} --update-frequency ${update} --seq-length ${seq_len}\
 --logdir logs/len_${seq_len}_chain_${num_chain}/ --baseline-lr-positive 10 --chain-lr-positive 10 --threshold ${threshold} \
 --chain-train-step ${chain_train_step} --baseline-train-step ${baseline_train_step} ${baseline} ${chained}
