import os
import argparse
from typing import List
from typing import Union
from datetime import datetime
import solver
from utils import functions
import joblib
import json
import numpy as np

from scipy.optimize import fsolve


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


def orthogonalize(U, V):

    def proj(u, v):
        """
        Return projection of v to u
        """
        return np.dot(v, u) / np.dot(u, u) * u

    U = V - np.sum(proj(U, V))
    return U, V

def func_max_overlap(x, T):
    return 1 / T * x * (1 - 1 / np.pi * np.arccos(x)) - np.sqrt(
        2 / np.pi) * (1 - x**2)


def main(input_dim: int, num_tasks: int, seq_length: int, cossim: float,vt: List[float],
         w_angle: float, lr_w1: float, lr_w2: float, lr_v: float, max_iters: List[int],
         simulation: int, ode: int, logdir: str, seeds: int, w_noise:float, v_noise:float, args: dict):

    ## Initialization:
    log_time = datetime.now().strftime("%Y%m%d%H%M%S.%f")
    log_folder = os.path.join(logdir, log_time)
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    with open(os.path.join(log_folder, 'args.json'), 'w') as f:
        json.dump(args, f)
    max_iters = np.array(max_iters)

    x_init = 0.9
    threshold = fsolve(func_max_overlap, x_init, args=(seq_length)) - 0.01

    for seed in range(seeds):

        _, WT = gram_schmidt(input_dim, num_tasks)
        WS = WT.copy()
        for i, w in enumerate(WS):
            w_rot = functions.control_VS(w, w_angle, False) * np.sqrt(input_dim)
            WS[i] = w_rot
        WS_copy = WS.copy()

        VT = np.array(vt)
        VT /= np.linalg.norm(VT)
        VS = functions.control_VS(VT, np.arccos(cossim))
        VS_copy = VS.copy()
        lr_ws = [lr_w1, lr_w2]
        if ode:
            result_ode = {'pretraining': None, 'composite': None}
            ode_solver = solver.HRLODESolver(VS=VS,
                                             VT=VT,
                                             WS=WS,
                                             WT=WT,
                                             lr_ws=lr_ws,
                                             lr_v=lr_v,
                                             seq_length=seq_length,
                                             N=input_dim,
                                             V_norm=0)

            ode_solver._init_params()

            for i in range(max_iters[0]):
                ode_solver._step1(update=False, DQ=None, DR=None)
                if abs(ode_solver.overlap_task[0]) >= threshold:
                    result_ode['pretraining'] = {
                        'iter': i,
                        'threshold': threshold,
                        'overlap_task': ode_solver.overlap_task,
                        'overlap_tilde': ode_solver.overlap
                    }
                    break
            print(ode_solver.overlap)

            for i in range(max_iters[1]):
                ode_solver._step2(update=False, DV=None, DQ=None, DR=None)
                if abs(ode_solver.overlap) >= threshold:
                    result_ode['composite'] = {
                        'iter': i,
                        'threshold': threshold,
                        'overlap_task': ode_solver.overlap_task,
                        'overlap_tilde': ode_solver.overlap
                    }
                    break
            print(ode_solver.overlap)

            joblib.dump(result_ode, os.path.join(log_folder, f'ode_{seed}.jl'))

        if simulation:
            result_sim = {'pretraining': None, 'composite': None}
            sim_solver = solver.simple_hrl_solver.CurriculumCompositionalTaskSimulator(
                input_dim=input_dim,
                seq_len=seq_length,
                num_task=num_tasks,
                identical=False,
                WT=WT,
                WS=WS_copy,
                VT=VT,
                VS=VS_copy,
                V_norm=0,
                w_noise=w_noise,
                v_noise=v_noise)
            sim_solver.lr = {'lr_w': lr_ws[0], 'lr_wc': lr_ws[1], 'lr_vc': lr_v}
            sim_solver.setup_train()
            sim_solver.multipleRLPerceptron.WS = sim_solver.WS
            sim_solver.multipleRLPerceptron.WT = sim_solver.WT
            sim_solver.multipleRLPerceptron.lr_w = sim_solver.lr['lr_w']
            for i in range(max_iters[0]):
                sim_solver.pretrain_step()
                if np.all(abs(sim_solver.overlap_task) >= threshold):
                    result_sim['pretraining'] = {
                        'iter': i,
                        'threshold': threshold,
                        'overlap_task': sim_solver.overlap_task,
                        'overlap_tilde': sim_solver.overlap
                    }
                    break
            for i in range(max_iters[1]):
                sim_solver.step()
                if np.all(abs(sim_solver.overlap) >= threshold):
                    result_sim['composite'] = {
                        'iter': i,
                        'threshold': threshold,
                        'overlap_task': sim_solver.overlap_task,
                        'overlap_tilde': sim_solver.overlap
                    }
                    break
            joblib.dump(result_sim, os.path.join(log_folder, f'sim_{seed}.jl'))


if __name__ == '__main__':

    def _helper_list_input(args_dict, key, type_func):
        return [type_func(item) for item in args_dict[key].split(',')]

    parser = argparse.ArgumentParser()
    ##Task parameters
    parser.add_argument("--input-dim", type=int, default=1000)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=4)
    parser.add_argument("--simulation", type=int, default=0)
    parser.add_argument("--ode", type=int, default=1)

    ##Training parameters
    parser.add_argument("--max-iters", help='delimited list input', type=str)
    parser.add_argument("--lr-w1", help='delimited list input', type=float)
    parser.add_argument("--lr-w2", help='delimited list input', type=float)
    parser.add_argument("--cossim", help='vs', type=float)
    parser.add_argument("--vt", help='vt', type=str, default='1,1')
    parser.add_argument("--w-angle", help='angle between WS and WT', type=float)
    parser.add_argument("--lr-v", type=float)
    parser.add_argument("--logdir", type=str, default='../hrl_ode_matrix')
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--w-noise", type=float, default=0.)
    parser.add_argument("--v-noise", type=float, default=0.)

    ##

    args = vars(parser.parse_args())

    args['max_iters'] = _helper_list_input(args, 'max_iters', int)
    #args['lr_ws'] = _helper_list_input(args, 'lr_ws', float)
    args['vt'] = _helper_list_input(args, 'vt', float)

    main(**args, args=args)