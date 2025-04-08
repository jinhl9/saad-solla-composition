"""
This is a script to extract training time for each experiment to find the best learning rate for each case
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib as jl
import json
import glob

def load_log(p):
    log =jl.load(p)
    args = json.load(open('/'.join(p.split('/')[:-1]) + '/args.json', 'r'))    
    return log, args

def get_training_time(p, k, lrw1, lrw2, lrv):
    log, args = load_log(p)
    num_tasks = args['num_tasks']
    seq_length = args['seq_length']
    cossim = args['cossim']
    w_angle = args['w_angle']
    pretrain_num_iter = 0
    composite_num_iter = 0
    num_iter = 0
    overlap_tilde = None
    if args['max_iters'][0] == 0 and num_tasks==k: ##Baseline case
        if log['composite'] is not None:
            threshold = log['composite']['threshold']
            num_iter = log['composite']['iter']
            overlap_tilde = log['composite']['overlap_tilde']

    elif num_tasks==k and \
    args['lr_w1']== lrw1 and args['lr_w2']== lrw2 and args['lr_v']== lrv:## Pretraining case
        if log['pretraining'] is not None:
            pretrain_threshold = log['pretraining']['threshold']
            pretrain_num_iter = log['pretraining']['iter']
            overlap = log['pretraining']['overlap_task']
        if log['composite'] is not None:
            composite_threshold = log['composite']['threshold']
            composite_num_iter = log['composite']['iter']
            overlap_tilde = log['composite']['overlap_tilde']
        
        num_iter = num_tasks * pretrain_num_iter + composite_num_iter
        
    
    return w_angle, seq_length, tuple(vs), num_iter, overlap_tilde

def lr_hyperparams_search(rootpath, T, K, init_w, pretrain):
    
    def _pretrain_iter(log):
        
        if log['pretraining'] is not None:
            pretrain_threshold = log['pretraining']['threshold']
            pretrain_num_iter = log['pretraining']['iter']
            overlap = log['pretraining']['overlap_task']
        
            return pretrain_num_iter
        return np.inf 
    
    def _composite_iter(log):
        
        if log['composite'] is not None:
            composite_threshold = log['composite']['threshold']
            composite_num_iter = log['composite']['iter']
            overlap_tilde = log['composite']['overlap_tilde']
        
            return composite_num_iter
        
        return np.inf
    
        
    results_dict={}
    for p in glob.glob(rootpath):
        log, args = load_log(p)
        num_tasks = args['num_tasks']
        seq_length = args['seq_length']
        w_angle = args['w_angle']
        if num_tasks==K and seq_length==T and w_angle==init_w:

            lr_w1= args['lr_w1']
            lr_w2= args['lr_w2']
            lr_v= args['lr_v']
            #vs = tuple(args['vs'])
            vs = args['cossim']
            pretrain_num_iter =  0
            if pretrain:
                pretrain_num_iter = _pretrain_iter(log)
                
            composite_num_iter = _composite_iter(log)
                
            num_iter = num_tasks * pretrain_num_iter + composite_num_iter
            
            if not vs in results_dict.keys():
                results_dict[vs] = {}
                
            results_dict[vs][(lr_w1, lr_w2, lr_v)] = [pretrain_num_iter,composite_num_iter, num_iter]
                
    best_dict= {}
    print(results_dict.keys())
    for k,v in results_dict.items():
        
        best_lr=list(v.keys())[np.argmin(np.array(list(v.values()))[:,-1])]
        best_iter = np.min(np.array(list(v.values()))[:,-1])
        assert best_iter == v[best_lr][-1]
        
        best_dict[k] = (v[best_lr], best_lr)    
    return results_dict, best_dict
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-rootpath', type=str)
    parser.add_argument('--curriculum-rootpath', type=str)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--K', type=int)
    parser.add_argument('--T', type = int)
    parser.add_argument('--W-init', type = float)

    args = parser.parse_args()
    
    baseline_hyperparam, best_baseline = lr_hyperparams_search(args.baseline_rootpath, args.T, args.K, args.W_init, False )
    curriculum_hyperparam, best_curriculum = lr_hyperparams_search(args.curriculum_rootpath, args.T, args.K, args.W_init, True )
    log_folder = f'{args.savedir}/K{args.K}T{args.T}Winit{args.W_init}'
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    jl.dump(baseline_hyperparam,os.path.join(log_folder,'baseline_hp_search.jl'))
    jl.dump(curriculum_hyperparam,os.path.join(log_folder,'curriculum_hp_search.jl'))
    print(log_folder)
    #search_space = list(list(baseline_hyperparam.values())[0].keys())
    result_dict = {}
    for k,v in best_baseline.items():
        best_curriculum_iter, curriculum_best_hp = best_curriculum[k]
        best_baseline_iter, baseline_best_hp = v

        iter_ratio = best_baseline_iter[-1]/best_curriculum_iter[-1]
        result_dict[k] = {'best_curriculum_iter': best_curriculum_iter, 'best_baseline_iter': best_baseline_iter,
                          'best_curriculum_hp': curriculum_best_hp, 'best_baseline_hp': baseline_best_hp,
                          'ratio': iter_ratio,}
                          #'search_space':search_space}
    
    jl.dump(result_dict,os.path.join(log_folder,'result_summary.jl'))
    


    

