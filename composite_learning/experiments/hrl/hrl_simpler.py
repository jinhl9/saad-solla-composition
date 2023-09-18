import os
import argparse
from typing import List
from typing import Union
from datetime import datetime
import solver
import joblib
import json
import numpy as np


def gram_schmidt(N, K):
    """
    Given the dimension space dimension N, generate K random vectors and its orthogonal spans
    """

    def proj(u, v):
        """
        Return projection of v to u
        """
        return np.dot(v, u) / np.dot(u, u) * u

    V = np.random.normal(loc=0., scale=1., size=(K, N))
    U = np.zeros_like(V)

    ## Initialise u1 to v1
    U[0] = V[0]

    ## Gram-schomidt process
    for k in range(1, K):
        projection_terms = [proj(U[i], V[k]) for i in range(k)]
        U[k] = V[k] - np.sum(projection_terms, axis=0)

    return V, U


def control_VS(VT, angle):
    dim = len(VT)
    VT_norm = VT / np.linalg.norm(VT)
    a = np.random.normal(loc=0., scale=1, size=(dim))
    b = np.random.normal(loc=0., scale=1, size=(dim))
    h = (b - a) - np.dot((b - a), VT_norm) * VT_norm
    v = np.cos(angle) * VT_norm + np.sin(angle) * h / np.linalg.norm(h)

    return v * 0.00001


def main(input_dim: int, num_tasks: int, seq_length: int, v_angle: float,
         vt_weights: List[float], lr_ws: List[float], lr_v: float,
         nums_iter: List[int], update_frequency: int, logdir: str, seeds: int,
         noise_scale: float, args: dict):

    log_time = datetime.now().strftime("%Y%m%d%H%M%S.%f")
    log_folder = os.path.join(logdir, log_time)
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    with open(os.path.join(log_folder, 'args.json'), 'w') as f:
        json.dump(args, f)
    nums_iter = np.array(nums_iter)

    for seed in range(seeds):
        _, WT_sim = gram_schmidt(input_dim, num_tasks)

        WS_sim = np.random.normal(loc=0., scale=1., size=(num_tasks, input_dim))

        WS_sim = np.divide(WS_sim * np.sqrt(input_dim),
                           np.linalg.norm(WS_sim, axis=1)[:, None])

        VT_sim = np.array(vt_weights)
        VS_sim = control_VS(VT_sim, v_angle)
        VS_sim /= np.linalg.norm(VS_sim)
        VS_ode = VS_sim.copy()
        VT_ode = VT_sim.copy()
        WS_ode = WS_sim.copy()
        WT_ode = WT_sim.copy()

        ode_solver = solver.HRLODESolver(VS=VS_ode,
                                         VT=VT_ode,
                                         WS=WS_ode,
                                         WT=WT_ode,
                                         lr_ws=lr_ws,
                                         lr_v=lr_v,
                                         seq_length=seq_length,
                                         N=input_dim)
        ode_solver.train(nums_iter, update_frequency=update_frequency)

        joblib.dump({'nid': ode_solver.history},
                    os.path.join(log_folder, f'ode_{seed}.jl'))

        sim = solver.simple_hrl_solver.CurriculumCompositionalTaskSimulator(
            input_dim=input_dim,
            seq_len=seq_length,
            num_task=num_tasks,
            identical=False,
            WT=WT_sim,
            WS=WS_sim,
            VT=VT_sim,
            VS=VS_sim)

        sim.train(num_iter=nums_iter,
                  update_frequency=update_frequency,
                  lr={
                      'lr_w': lr_ws[0],
                      'lr_wc': lr_ws[1],
                      'lr_vc': lr_v
                  })

        joblib.dump({'nid': sim.history},
                    os.path.join(log_folder, f'sim_{seed}.jl'))


if __name__ == '__main__':

    def _helper_list_input(args_dict, key, type_func):
        return [type_func(item) for item in args_dict[key].split(',')]

    parser = argparse.ArgumentParser()
    ##Task parameters
    parser.add_argument("--input-dim", type=int, default=1000)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=4)
    parser.add_argument("--vt-weights", help='initial VT', type=str)
    parser.add_argument("--noise-scale", type=float, help='added noise to VT')

    ##Training parameters
    parser.add_argument("--nums-iter", help='delimited list input', type=str)
    parser.add_argument("--lr-ws", help='delimited list input', type=str)
    parser.add_argument("--v-angle", help='angle between VS and VT', type=float)
    parser.add_argument("--lr-v", type=float)
    parser.add_argument("--update-frequency", type=int, default=1000)
    parser.add_argument("--logdir", type=str, default='../hrl_ode_matrix')
    parser.add_argument("--seeds", type=int, default=1)

    args = vars(parser.parse_args())

    args['nums_iter'] = _helper_list_input(args, 'nums_iter', int)
    args['lr_ws'] = _helper_list_input(args, 'lr_ws', float)
    args['vt_weights'] = _helper_list_input(args, 'vt_weights', float)

    main(**args, args=args)