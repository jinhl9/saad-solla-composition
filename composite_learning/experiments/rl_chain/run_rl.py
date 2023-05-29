import os
import argparse
import copy
from datetime import datetime
import data
import network
import solver
import time


def main(baseline: bool, chained: bool, input_dim: int, seq_length: int,
         num_chain: int, baseline_lr_positive: float, chain_lr_positive: float,
         chain_train_step: int, baseline_train_step: int, logdir: str,
         save_frequency: int, update_frequency: int, threshold: float):
    print(baseline, chained)

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
    teachers, students, naive_students = _define_models(num_chain, input_dim)
    single_data_loader = data.TransientRLTask(batch_size=1,
                                              seq_len=[seq_length],
                                              input_dim=input_dim)
    total_data_loader = data.TransientRLTask(batch_size=1,
                                             seq_len=[seq_length] * num_chain,
                                             input_dim=input_dim)
    """
    Train baseline model
    """
    if baseline:
        print('Baseline start')
        t0 = time.time()
        baseline = solver.MultistepRLPerceptronSolver(
            criterion=None,
            logdir=os.path.join(logdir, log_time, 'baseline'),
            optimizer_type='sgd',
            lr=None,
            lr_positive=baseline_lr_positive,
            lr_negative=None,
            weight_decay=0,
            teacher_network=teachers,
            student_network=naive_students)
        print(f'Time to initialize solver: {time.time()-t0}')
        t0 = time.time()
        baseline.train(data_loader=total_data_loader,
                       num_iter=baseline_train_step,
                       save_last=True,
                       save_frequency=save_frequency,
                       update_frequency=update_frequency)
        print(f'Time to finish training baseline: {time.time()-t0}')

    if chained:
        """
        Train chained models
        """
        print('Start chain model')
        print(len(teachers), len(students))
        chained_students = []
        for i in range(num_chain):
            t0 = time.time()
            single_chain = solver.MultistepRLPerceptronSolver(
                criterion=None,
                logdir=os.path.join(logdir, log_time, f'chain{i}'),
                optimizer_type='sgd',
                lr=None,
                lr_positive=chain_lr_positive,
                lr_negative=None,
                weight_decay=0,
                teacher_network=[teachers[i]],
                student_network=[students[i]],
                use_threshold=True,
                threshold=threshold)
            print(f'Time to initialzie one chain solver: {t0-time.time()}')
            t0 = time.time()
            single_chain.train(data_loader=single_data_loader,
                               num_iter=chain_train_step,
                               save_last=True,
                               save_frequency=save_frequency,
                               update_frequency=update_frequency)
            print(f'Time to train one chain solver: {t0-time.time()}')
            t0 = time.time()
            if i != (num_chain - 1):
                chained_students.append(single_chain.student_network[0])
            else:
                chained_students += single_chain.student_network
            print(f'Time to append one chain solver: {t0-time.time()}')
        """
        t0 = time.time()
        chained_solver = solver.MultistepRLPerceptronSolver(
            criterion=None,
            logdir=os.path.join(logdir, log_time, f'total_chain'),
            optimizer_type='sgd',
            lr=None,
            lr_positive=chain_lr_positive,
            lr_negative=None,
            weight_decay=0,
            teacher_network=teachers,
            student_network=chained_students,
            use_threshold=True,
            threshold=threshold)
        print(f'Time to initialize total solver: {t0-time.time()}')
        t0 = time.time()
        chained_solver.train(data_loader=total_data_loader,
                             num_iter=chain_train_step,
                             save_last=True,
                             save_frequency=save_frequency,
                             update_frequency=update_frequency)
        print(f'Time to train a total solver: {t0-time.time()}')
        """


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ##Experiment parameter
    parser.add_argument('--baseline', action=argparse.BooleanOptionalAction)
    parser.add_argument('--chained', action=argparse.BooleanOptionalAction)
    ##Task parameter
    parser.add_argument('--input-dim', type=int, default=100)
    parser.add_argument('--seq-length', type=int, default=3)
    parser.add_argument('--num-chain', type=int, default=2)
    ##Training parameter
    parser.add_argument('--chain-train-step', type=int, default=300)
    parser.add_argument('--baseline-train-step', type=int, default=300)
    parser.add_argument('--save-frequency', type=int, default=None)
    parser.add_argument('--update-frequency', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='log/')
    parser.add_argument('--baseline-lr-positive', type=float, default=10.)
    parser.add_argument('--chain-lr-positive', type=float, default=10.)
    parser.add_argument('--threshold', type=float, default=0.90)

    args = parser.parse_args()

    main(**vars(args))