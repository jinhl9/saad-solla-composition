#! /bin/bash
for lenT in 4 6 8 10; do
for winit in 1.57; do
python experiments/hrl/lr_search_result.py --baseline-rootpath "hrl_ode_logs/260124_baseline_k4/*/ode*" --curriculum-rootpath "hrl_ode_logs/260124_curriculum_k4/*/ode*" --savedir "hrl_ode_logs/lr_param_search/260124_k4/" --K 4 --T ${lenT} --W-init ${winit}
done
done