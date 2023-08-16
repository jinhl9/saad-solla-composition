import os
import argparse
from typing import List
from datetime import datetime
import data
import network
import solver
import joblib
import json

import torch
import numpy as np

def load_trained_networks(params:dict, weights:dict):

    students = []
    teachers = []

    for k in range(params['num_tasks']):
        student = network.ContinuousStudent(input_dimension=params['input_dim'],
                                        hidden_dimensions=[1],
                                        nonlinearity='sign',
                                        normalize=True)
        
        teacher = network.ContinuousTeacher(input_dimension=params['input_dim'],
                                            hidden_dimensions=[1],
                                            nonlinearity='sign',
                                            standardize=True,
                                            normalize=True)
        
        student.layers[0].weight.data = weights[k]['WS']
        teacher.layers[0].weight.data = weights[k]['WT']
        students.append(student)
        teachers.append(teacher)
    student_c = network.ContextStudent(input_dimension=params['num_tasks'],
                                    hidden_dimensions=[1],
                                    nonlinearity='sign',
                                    initialisation_std = 0.001,
                                    normalize=False)
    
    student_c.layers[0].weight.data = weights['vs']

    return students, teachers, student_c


def main(logpath:str, perturbation_level:float, num_iter:int, lr_w:float, lr_v:float, update_frequency: int, seeds:int):
    ##Load previously trained experiment
    assert os.path.isfile(logpath.replace('simlator_', 'simulator_weights')), 'Weights file not existing'
    
    rootpath= logpath.split('simulator')[0]
    sim_seed=logpath.split('simulator_')[1].split('.jl')[0]
    log_time = datetime.now().strftime("%Y%m%d%H%M%S%f")
    logdir = os.path.join(rootpath, log_time)
    os.mkdir(logdir)
    with open(os.path.join(logdir, f'perturbation_args.json'), 'w') as f:
            json.dump(args, f)

    params=json.load(open(os.path.join(rootpath, 'args.json'),'r'))
    weights=joblib.load(logpath.replace('.jl', '_weights.jl'))
    perturbed_context = torch.FloatTensor([[1+perturbation_level, 1-perturbation_level, 1+perturbation_level, 1-perturbation_level]])

    for seed in range(seeds):
        students, teachers, student_c = load_trained_networks(params, weights)
        teacher_c = network.ContextTeacher(input_dimension=params['num_tasks'],
                                       hidden_dimensions=[1],
                                       nonlinearity='sign',
                                       normalize=False)
        teacher_c.layers[0].weight.data = perturbed_context
        dataloader = data.iid.TransientRLTask(batch_size = 1, seq_len = [params['seq_length']]*params['num_tasks'], input_dim = params['input_dim'], identical = params['identical'])
        simulator = solver.TwoPhaseContextSolver(teachers=teachers, students = students, context_teacher=teacher_c, context_student=student_c, dataloaders = [dataloader, dataloader, dataloader], logdir = None, identical=params['identical'])
        simulator.train(nums_iter = [0,num_iter], lrs = [(0, 0),(lr_w, lr_v)], update_frequency=update_frequency)

        joblib.dump(simulator.history,  os.path.join(logdir, f'perturbation_sim{sim_seed}_{seed}.jl'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Experiment params
    parser.add_argument('--logpath', default = None, type= str, help='Path to the trained model weights to use for experiment')
    parser.add_argument('--perturbation-level', default = 0.1, type = float)
    parser.add_argument('--seeds', type = int)

    ## Training params
    parser.add_argument('--num-iter', type=int)
    parser.add_argument('--update-frequency', type=int)
    parser.add_argument('--lr-v', type = float)
    parser.add_argument('--lr-w', type = float)
    

    args = vars(parser.parse_args())
    
    main(**args)
