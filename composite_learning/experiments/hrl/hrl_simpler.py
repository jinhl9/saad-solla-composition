import os
import argparse
from typing import List
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
    a = np.random.normal(loc=0., scale=1., size=(dim))
    b = np.random.normal(loc=0., scale=1., size=(dim))
    h = (b - a) - np.dot((b - a), VT_norm) * VT_norm
    v = np.cos(angle) * VT_norm + np.sin(angle) * h / np.linalg.norm(h)

    return v


def main(input_dim: int, num_tasks: int, seq_length: int, v_angle: float,
         lr_ws: List[float], lr_v: float, nums_iter: List[int],
         update_frequency: int, logdir: str, seeds: int, args: dict):

    log_time = datetime.now().strftime("%Y%m%d%H%M%S.%f")
    log_folder = os.path.join(logdir, log_time)
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    with open(os.path.join(log_folder, 'args.json'), 'w') as f:
        json.dump(args, f)
    nums_iter = np.array(nums_iter)
    for seed in range(seeds):
        _, WT_id = gram_schmidt(input_dim, num_tasks)
        WS_id = np.random.normal(loc=0., scale=1., size=(num_tasks, input_dim))
        VT_id = np.random.normal(loc=0., scale=1., size=(num_tasks))
        VS_id = control_VS(VT_id, v_angle)

        VS_nid = VS_id.copy()
        VT_nid = VT_id.copy()
        WS_nid = WS_id.copy()
        WT_nid = WT_id.copy()

        ode_solver_id = solver.HRLODESolverIdentical(VS=VS_id,
                                                     VT=VT_id,
                                                     WS=WS_id,
                                                     WT=WT_id,
                                                     lr_ws=lr_ws,
                                                     lr_v=lr_v,
                                                     seq_length=seq_length,
                                                     N=input_dim)
        ode_solver_id.train(nums_iter, update_frequency=update_frequency)

        ode_solver_nid = solver.HRLODESolver(VS=VS_nid,
                                             VT=VT_nid,
                                             WS=WS_nid,
                                             WT=WT_nid,
                                             lr_ws=lr_ws,
                                             lr_v=lr_v,
                                             seq_length=seq_length,
                                             N=input_dim)
        ode_solver_nid.train(nums_iter, update_frequency=update_frequency)

        joblib.dump({
            'id': ode_solver_id.history,
            'nid': ode_solver_nid.history
        }, os.path.join(log_folder, f'ode_{seed}.jl'))


if __name__ == '__main__':

    def _helper_list_input(args_dict, key, type_func):
        return [type_func(item) for item in args_dict[key].split(',')]

    parser = argparse.ArgumentParser()
    ##Task parameters
    parser.add_argument("--input-dim", type=int, default=1000)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=4)

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

    main(**args, args=args)