#! /bin/bash
echo "Launch chained RL vs. baseline exp"
step=${@:$#}
echo "${step}"
for var in "${@:1:$#-1}"; do
echo "Experiment for sequence length ${var}"
python experiments/rl_chain/run_rl.py --update-frequency 200 --seq-length ${var} --logdir logs/len_${var}/ --lr-positive 10 --train-step ${step}
done