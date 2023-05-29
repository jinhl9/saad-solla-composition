import os
import argparse
import copy
from datetime import datetime
import data
import network
import solver


def main(input_dim: int, seq_length: int, num_chain: int, lr_positive: float,
         train_step: int, logdir: str, save_frequency: int):

    def _define_models(model_num, input_dim):
        teachers = []
        students = []

        for i in range(model_num):
            teachers.append(
                network.ContinuousTeacher(input_dimension=input_dim,
                                          hidden_dimensions=[1],
                                          nonlinearity='sign'))
            students.append(
                network.ContinuousStudent(input_dimension=input_dim,
                                          hidden_dimensions=[1],
                                          nonlinearity='sign'))

        return teachers, students, copy.deepcopy(students)

    """
    Define models and data loader
    """

    log_time = datetime.now().strftime("%Y%m%d%H%M%S")
    teachers, students, naive_students = _define_models(num_chain * 2,
                                                        input_dim)
    single_data_loader = data.TransientRLTask(batch_size=1,
                                              seq_len=[seq_length, 1],
                                              input_dim=input_dim)
    total_data_loader = data.TransientRLTask(batch_size=1,
                                             seq_len=[seq_length] * num_chain +
                                             [1],
                                             input_dim=input_dim)
    single_chain_solvers = []
    """
    Train chained models
    """
    for i in range(num_chain):
        single_chain = solver.ChainedRLPerceptronSolver(
            criterion=None,
            logdir=os.path.join(logdir, log_time, f'chain{i}'),
            optimizer_type='sgd',
            lr=None,
            lr_positive=lr_positive,
            lr_negative=None,
            weight_decay=0,
            teacher_network=teachers[i * 2:i * 2 + 2],
            student_network=students[i * 2:i * 2 + 2])

        single_chain.train(data_loader=single_data_loader,
                           num_iter=train_step,
                           save_last=True,
                           save_frequency=save_frequency)
        single_chain_solvers.append(single_chain)
    """
    Train a baseline model
    """
    baseline = solver.ChainedRLPerceptronSolver(criterion=None,
                                                logdir=os.path.join(
                                                    logdir, log_time,
                                                    'baseline'),
                                                optimizer_type='sgd',
                                                lr=None,
                                                lr_positive=lr_positive,
                                                lr_negative=None,
                                                weight_decay=0,
                                                teacher_network=teachers,
                                                student_network=naive_students)

    baseline.train(data_loader=total_data_loader,
                   num_iter=train_step * num_chain,
                   save_last=True,
                   save_frequency=save_frequency)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ##Task parameter
    parser.add_argument('--input-dim', type=int, default=100)
    parser.add_argument('--seq-length', type=int, default=3)
    parser.add_argument('--num-chain', type=int, default=2)
    ##Training parameter
    parser.add_argument('--train-step', type=int, default=300)
    parser.add_argument('--save-frequency', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='log/')
    parser.add_argument('--lr-positive', type=float, default=10.)

    args = parser.parse_args()

    main(**vars(args))