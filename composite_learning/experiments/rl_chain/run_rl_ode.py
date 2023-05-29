import os
import argparse
import joblib
from datetime import datetime

from solver import ode


def main(baseline: bool, chained: bool, seq_length: int, num_chain: int,
         delta: float, train_per_chain: int, total_iter: int,
         lr_positive: float, lr_negative: float, update_frequency: int,
         logdir: str):

    ## Make log folder
    log_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_folder = os.path.join(logdir, log_time)
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    ode_solver = ode.MultistepRLPerceptronODESolver(
        lr_positive=lr_positive,
        lr_negative=lr_negative,
        seq_length=seq_length,
        num_chain=num_chain,
        train_per_chain=train_per_chain,
        delta=delta,
        update_frequency=update_frequency,
        threshold=True)
    ode_solver.init_params()
    joblib.dump({'teachers': ode_solver.teachers},
                os.path.join(log_folder, 'teachers.jl'))
    if baseline:
        ## Simulate baseline

        ode_solver.train(total_iter, 'baseline')

        joblib.dump(ode_solver.history, os.path.join(log_folder, 'baseline.jl'))

    ##Simulate chained
    if chained:
        ode_solver.train(train_per_chain * num_chain, 'chained')
        joblib.dump(ode_solver.history, os.path.join(log_folder, 'chained.jl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ##Experiment parameters
    parser.add_argument("--baseline", action=argparse.BooleanOptionalAction)
    parser.add_argument("--chained", action=argparse.BooleanOptionalAction)
    ##Task parameters
    parser.add_argument("--seq-length", type=int, default=3)
    parser.add_argument("--num-chain", type=int, default=2)
    parser.add_argument("--delta", type=float, default=0.01)

    ##Training parameters
    parser.add_argument("--train-per-chain", type=int, default=1000)
    parser.add_argument("--total-iter", type=int, default=10000)
    parser.add_argument("--lr-positive", type=float, default=10.)
    parser.add_argument("--lr-negative", type=float, default=0.)
    parser.add_argument("--update-frequency", type=int, default=1000)
    parser.add_argument("--logdir", type=str, default='../ode_logs')

    args = parser.parse_args()

    main(**vars(args))