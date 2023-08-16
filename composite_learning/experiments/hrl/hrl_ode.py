import os
import argparse
from typing import List
from datetime import datetime
import data
import network
import solver
import joblib
import json


def initialize_nets(input_dim: int, num_tasks: int):
    teachers = []
    students = []
    for i in range(num_tasks):
        teacher = network.ContinuousTeacher(input_dimension=input_dim,
                                            hidden_dimensions=[1],
                                            nonlinearity='sign',
                                            standardize=True,
                                            normalize=True)
        student = network.ContinuousStudent(input_dimension=input_dim,
                                            hidden_dimensions=[1],
                                            nonlinearity='sign',
                                            normalize=True)
        teachers.append(teacher)
        students.append(student)

    teacher_c = network.ContextTeacher(input_dimension=num_tasks,
                                       hidden_dimensions=[1],
                                       weights = [1.],
                                       nonlinearity='sign',
                                       normalize=False)
    student_c = network.ContextStudent(input_dimension=num_tasks,
                                       hidden_dimensions=[1],
                                       nonlinearity='sign',
                                       initialisation_std = 0.001,
                                       normalize=False)

    return teachers, students, teacher_c, student_c


def main(input_dim: int, num_tasks: int, seq_length: int, lr_ws: List[float],
         lr_v: float, nums_iter: List[int], update_frequency: int, logdir: str, seeds:int, identical:bool,
         args: dict):

    ## Make log folder
    log_time = datetime.now().strftime("%Y%m%d%H%M%S.%f")
    log_folder = os.path.join(logdir, log_time)
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    with open(os.path.join(log_folder, 'args.json'), 'w') as f:
        json.dump(args, f)

    
    for seed in range(seeds):

        teachers, students, context_teacher, context_student = initialize_nets(
            input_dim, num_tasks)
        VS = context_student.layers[0].weight.data.numpy()[0]
        VT = context_teacher.layers[0].weight.data.numpy()[0]
        if identical:
            ode_solver = solver.HierarchicalODESolverIdentical(VS=VS,
                                                    VT=VT,
                                                    students=students,
                                                    teachers=teachers,
                                                    lr_ws=lr_ws,
                                                    lr_v=lr_v,
                                                    seq_length=seq_length,
                                                    N=input_dim)
        else:
            ode_solver = solver.HierarchicalODESolver(VS=VS,
                                                    VT=VT,
                                                    students=students,
                                                    teachers=teachers,
                                                    lr_ws=lr_ws,
                                                    lr_v=lr_v,
                                                    seq_length=seq_length,
                                                    N=input_dim)

        ode_solver.train(nums_iter=nums_iter, update_frequency=update_frequency)
        joblib.dump(ode_solver.history, os.path.join(log_folder, f'ode_{seed}.jl'))


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
    parser.add_argument("--lr-v", type=float)
    parser.add_argument("--update-frequency", type=int, default=1000)
    parser.add_argument("--logdir", type=str, default='../hrl_ode_logs')
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument('--identical', action = 'store_true')

    args = vars(parser.parse_args())

    args['nums_iter'] = _helper_list_input(args, 'nums_iter', int)
    args['lr_ws'] = _helper_list_input(args, 'lr_ws', float)

    main(**args, args=args)
