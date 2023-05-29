import os
import math
from dataclasses import dataclass
import logging
import joblib
from typing import List
from typing import Dict
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import time
import multiprocessing as mp

from solver import base
import network


@dataclass(kw_only=True)
class RLSolver(base.BaseSolver):
    teacher_network: network.BaseTeacher
    student_network: network.BaseStudent
    lr: float
    weight_decay: float
    optimizer_type: str

    def __post_init__(self):
        self._train = False
        self._train_args = {}

    def _setup_history(self):
        self.history = {
            'loss': [],
            'prediction': [],
            'true': [],
            'exp_reward_rate': []
        }

    def _history_update(self, **kwargs):
        for k, v in kwargs.items():
            self.history[k].append(v)

    def inference(self, x):
        return self.student_network(x)

    def _setup_train(self, **kwargs):
        self._train = True
        self._setup_history()
        if self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                params=self.student_network.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay)

    def _save_model(self, student=True, teacher=True, ckp_tag=''):
        if student:
            torch.save(self.student_network.state_dict(),
                       os.path.join(self.logdir, f'student{ckp_tag}.pt'))
        if teacher:
            torch.save(self.teacher_network.state_dict(),
                       os.path.join(self.logdir, f'teacher{ckp_tag}.pt'))

    def train(self,
              data_loader=None,
              num_iter=10,
              save_last=False,
              update_frequency=None,
              save_frequency=None):
        t0 = time.time()
        self._train_args['data_loader'] = data_loader
        self._train_args['num_iter'] = num_iter
        self._train_args['save_last'] = save_last
        self._train_args['update_frequency'] = update_frequency
        self._train_args['save_last'] = save_frequency
        self._setup_train()
        self.logger.info(f'num_iter: {num_iter}')
        for i in range(num_iter):
            if not self._train:
                print(f'Reached threshold:{i}')
                break
            x = data_loader.get_batch()['x']
            if update_frequency is not None and i % update_frequency == 0:
                self._update = True
            loss = self._step(x)
            self._update = False
            if loss is None:
                self.logger.info('invalid batch')
            if save_frequency is not None and i % save_frequency == 0 and (i !=
                                                                           0):
                self._save_model(student=True,
                                 teacher=False,
                                 ckp_tag=f'_{i:07d}')
        self._train = False
        print(f'Time to train solver: {t0-time.time()}')
        t0 = time.time()
        if save_last:
            self._save_model(student=True, teacher=True, ckp_tag='_last')
            joblib.dump(self.history, os.path.join(self.logdir, 'history.jl'))
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
        print(f'Time to final save solver: {t0-time.time()}')

    def _step(self, x: torch.Tensor):
        """
        Make prediction, compute loss and update for an episode. 
        Args:
            x: batched input of size B * seq_len * input_dim
        """
        self.optimizer.zero_grad()
        teacher_output = self.teacher_network(x)
        student_output = self.student_network(x)
        correct = torch.all(teacher_output * student_output > 0,
                            dim=1).squeeze()
        loss = self.criterion(student_output[correct], teacher_output[correct])
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        self._history_update(loss=loss,
                             prediction=student_output.detach(),
                             true=teacher_output.detach(),
                             exp_reward_rate=correct.detach().sum() /
                             len(correct))

        return loss


@dataclass(kw_only=True)
class RLPerceptronSolver(RLSolver):
    teacher_network: network.BaseTeacher
    student_network: network.BaseStudent
    lr_positive: Union[float, None]
    lr_negative: Union[float, None]

    def __post_init__(self):
        self._train = False
        self.D = self.teacher_network.layers[0].weight.shape[1]
        self._train_args = {}

    def _setup_train(self, **kwargs):
        self._train = True
        self._setup_history()

    def _setup_history(self):
        self.history = {'QT': [], 'QS': [], 'R': [], 'exp_reward': []}

    def _step(self, x: torch.Tensor):
        """
        Make prediction, compute loss and update for an episode. 
        Args:
            x: a episode dictionary of each task state with seq_len * input_dim
        """
        self.T = len(x)
        teacher_output = self.teacher_network(x).squeeze()
        student_output = self.student_network(x).squeeze()
        if torch.all(teacher_output == student_output):
            w_update = self.lr_positive / math.sqrt(
                self.D) * (1 / self.T * student_output @ x).unsqueeze(0)
        else:
            w_update = torch.zeros_like(
                self.student_network.layers[0].weight.data)
        self.student_network.layers[0].weight.data += w_update

        with torch.no_grad():
            R = (self.teacher_network.layers[0].weight.data
                 @ self.student_network.layers[0].weight.data.T) / self.D
            QS = (self.student_network.layers[0].weight.data
                  @ self.student_network.layers[0].weight.data.T) / self.D
            QT = (self.teacher_network.layers[0].weight.data
                  @ self.teacher_network.layers[0].weight.data.T) / self.D

        exp_reward = torch.pow(
            1 - (1 / torch.pi) * torch.acos(R /
                                            (torch.sqrt(QS) * torch.sqrt(QT))),
            self.T)
        self._history_update(QS=QS.item(),
                             QT=QT.item(),
                             R=R.item(),
                             exp_reward=exp_reward.item())

        return exp_reward


@dataclass(kw_only=True)
class MultistepRLPerceptronSolver(RLSolver):
    teacher_network: List[network.BaseTeacher]
    student_network: List[network.BaseStudent]
    lr_positive: Union[float, None]
    lr_negative: Union[float, None]
    use_threshold: bool = False
    threshold: float = 0.9
    num_worker: int = 4

    def __post_init__(self):
        self._setup_logger()
        self.logger.info(f'solver_args:{self.__dict__}')
        self._train = False
        self.D = self.teacher_network[0].layers[0].weight.shape[1]
        self._train_args = {}

    def _setup_train(self, **kwargs):
        t0 = time.time()
        self._train = True
        self.update_count = 0
        self._setup_history()
        self.logger.info(f'train_args:{self._train_args}')
        print(f'Time to set up training: {t0-time.time()}')

    def _setup_history(self):
        self.history = {'exp_reward': []}

    def train(self,
              data_loader=None,
              num_iter=10,
              save_last=False,
              update_frequency=None,
              save_frequency=None):

        self._setup_train()

        self._train_args['data_loader'] = data_loader
        self._train_args['num_iter'] = num_iter
        self._train_args['save_last'] = save_last
        self._train_args['update_frequency'] = update_frequency
        self._train_args['save_last'] = save_frequency

        self.task_steps = data_loader.seq_len
        self.num_tasks = len(data_loader.seq_len)
        t0 = time.time()
        for i in range(num_iter):
            self._update_history = False
            if not self._train:
                break
            if update_frequency is not None and i % update_frequency == 0:
                self._update_history = True
            x = data_loader.get_batch()['x']
            self._step(x)
            if save_frequency is not None and (
                    i + 1) % save_frequency == 0 and save_frequency:
                self._save_model(student=True,
                                 teacher=False,
                                 ckp_tag=f'_{i:07d}')
        self.logger.info(f'Training time: {time.time()-t0}')
        if save_last:
            self._save_model(student=True, teacher=True, ckp_tag=f'_last')
            joblib.dump(self.history, os.path.join(self.logdir, 'history.jl'))
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()

    def _single_task_inference(self, x: torch.Tensor, index: int):
        return torch.stack(
            [self.teacher_network[index](x), self.student_network[index](x)])

    def _single_network_update(self, x: torch.Tensor, y: torch.Tensor,
                               index: int):
        w_update = self.lr_positive / math.sqrt(self.D) * (
            1 / self.task_steps[index] *
            y[:,
              sum(self.task_steps[:index]):sum(self.task_steps[:index + 1])]
            @ x[index]).squeeze(dim=0)
        self.student_network[index].layers[0].weight.data += w_update

    def _save_model(self, student=True, teacher=True, ckp_tag=''):

        def _save_single_model(model, model_name='model'):
            torch.save(model.state_dict(),
                       os.path.join(self.logdir, f'{model_name}.pt'))

        if student:
            for i, net in enumerate(self.student_network):
                _save_single_model(net, f'student_{i}{ckp_tag}')
        if teacher:
            for i, net in enumerate(self.teacher_network):
                _save_single_model(net, f'teacher_{i}{ckp_tag}')

    def _step(self, x: List[torch.Tensor]):
        with torch.no_grad():
            outputs = []
            for i in range(self.num_tasks):
                outputs.append(self._single_task_inference(x[i], i))
            outputs = torch.cat(outputs, dim=2).squeeze(dim=-1)
            teacher_output = outputs[0]
            student_output = outputs[1]
            if torch.all(teacher_output == student_output, dim=1):
                for i in range(self.num_tasks):
                    self._single_network_update(x, student_output, i)
            if self._update_history:
                cum_exp = 1.
                for m, i in enumerate(self.task_steps):
                    R = (self.teacher_network[m].layers[0].weight.data @ self.
                         student_network[m].layers[0].weight.data.T) / self.D
                    QS = (self.student_network[m].layers[0].weight.data @ self.
                          student_network[m].layers[0].weight.data.T) / self.D
                    QT = (self.teacher_network[m].layers[0].weight.data @ self.
                          teacher_network[m].layers[0].weight.data.T) / self.D
                    n = m + 1
                    exp_reward = 1 - (1 / torch.pi) * torch.acos(
                        R / (torch.sqrt(QS) * torch.sqrt(QT)))
                    cum_exp *= np.power(exp_reward.item(), i)

                    setattr(self, f'R{n}', R.item())
                    setattr(self, f'QS{n}', QS.item())
                    setattr(self, f'exp_reward{n}', exp_reward.item())

                    if f'R{n}' in self.history.keys():
                        self.history[f'R{n}'].append(R.item())
                    else:
                        self.history[f'R{n}'] = []
                    if f'QS{n}' in self.history.keys():
                        self.history[f'QS{n}'].append(QS.item())
                    else:
                        self.history[f'QS{n}'] = []
                    if f'exp_reward{n}' in self.history.keys():
                        self.history[f'exp_reward{n}'].append(exp_reward.item())
                    else:
                        self.history[f'exp_reward{n}'] = []
                setattr(self, 'exp_reward', cum_exp)
                self.history[f'exp_reward'].append(cum_exp)
                self.history['num_update'] = self.update_count
        if self.use_threshold and self.exp_reward1 >= self.threshold:
            self._train = False


@dataclass(kw_only=True)
class ChainedRLPerceptronSolver(RLSolver):
    teacher_network: List[network.BaseTeacher]
    student_network: List[network.BaseStudent]
    lr_positive: Union[float, None]
    lr_negative: Union[float, None]
    use_threshold: bool = False
    threshold: float = 0.9

    def __post_init__(self):
        t0 = time.time()
        self._setup_logger()
        self.logger.info(f'solver_args:{self.__dict__}')
        self._train = False
        self.D = self.teacher_network[0].layers[0].weight.shape[1]
        self._train_args = {}
        print(f'Time to post init solver: {t0-time.time()}')

    def _setup_train(self, **kwargs):
        t0 = time.time()
        self._train = True
        self._update = False
        self.update_count = 0
        self._setup_history()
        self.logger.info(f'train_args:{self._train_args}')
        print(f'Time to set up trainingr: {t0-time.time()}')

    def _setup_history(self):
        self.history = {'exp_reward': []}

    def train(self,
              data_loader=None,
              num_iter=10,
              save_last=False,
              update_frequency=None,
              save_frequency=None):

        self._setup_train()

        self._train_args['data_loader'] = data_loader
        self._train_args['num_iter'] = num_iter
        self._train_args['save_last'] = save_last
        self._train_args['update_frequency'] = update_frequency
        self._train_args['save_last'] = save_frequency

        self.task_steps = data_loader.seq_len
        self.num_tasks = len(data_loader.seq_len)

        t0 = time.time()

        for i in range(num_iter):
            self._update_history = False
            if not self._train:
                break
            if update_frequency is not None and i % update_frequency == 0:
                self._update_history = True
            x = data_loader.get_batch()['x']
            self._step(x)
            if save_frequency is not None and (
                    i + 1) % save_frequency == 0 and save_frequency:
                self._save_model(student=True,
                                 teacher=False,
                                 ckp_tag=f'_{i:07d}')
        self.logger.info(f'Training time: {time.time()-t0}')
        if save_last:
            self._save_model(student=True, teacher=True, ckp_tag=f'_last')
            joblib.dump(self.history, os.path.join(self.logdir, 'history.jl'))
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()

    def _inference(self, x: Dict[str, torch.Tensor]):
        teacher_output = torch.tensor([])
        student_output = torch.tensor([])
        for i, v in enumerate(x):
            teacher_output = torch.cat(
                [teacher_output, self.teacher_network[i](v).squeeze(dim=-1)],
                dim=1)
            student_output = torch.cat(
                [student_output, self.student_network[i](v).squeeze(dim=-1)],
                dim=1)
        return teacher_output, student_output

    def _save_model(self, student=True, teacher=True, ckp_tag=''):

        def _save_single_model(model, model_name='model'):
            torch.save(model.state_dict(),
                       os.path.join(self.logdir, f'{model_name}.pt'))

        if student:
            for i, net in enumerate(self.student_network):
                _save_single_model(net, f'student_{i}{ckp_tag}')
        if teacher:
            for i, net in enumerate(self.teacher_network):
                _save_single_model(net, f'teacher_{i}{ckp_tag}')

    def _step(self, x: Dict[str, torch.Tensor]):
        """
        Make prediction, compute loss and update for an episode. 
        Args:
            x: a episode seq_len * input_dim
        """
        teacher_output = torch.tensor([])
        student_output = torch.tensor([])
        for i, v in enumerate(x):
            teacher_output = torch.cat(
                [teacher_output, self.teacher_network[i](v).squeeze(dim=-1)],
                dim=1)
            student_output = torch.cat(
                [student_output, self.student_network[i](v).squeeze(dim=-1)],
                dim=1)
        if torch.all(teacher_output == student_output):
            self.update_count += 1
            for i, v in enumerate(x):
                w_update = self.lr_positive / math.sqrt(self.D) * (
                    1 / self.task_steps[i] *
                    student_output[:, sum(self.T[:i]):sum(self.T[:i + 1])]
                    @ v).squeeze(dim=0)
                self.student_network[i].layers[0].weight.data += w_update
        if self._update:
            with torch.no_grad():
                cum_exp = 1.
                for m, i in enumerate(self.task_steps):
                    R = (self.teacher_network[m].layers[0].weight.data @ self.
                         student_network[m].layers[0].weight.data.T) / self.D
                    QS = (self.student_network[m].layers[0].weight.data @ self.
                          student_network[m].layers[0].weight.data.T) / self.D
                    QT = (self.teacher_network[m].layers[0].weight.data @ self.
                          teacher_network[m].layers[0].weight.data.T) / self.D
                    n = m + 1
                    exp_reward = 1 - (1 / torch.pi) * torch.acos(
                        R / (torch.sqrt(QS) * torch.sqrt(QT)))
                    cum_exp *= np.power(exp_reward.item(), i)

                    setattr(self, f'R{n}', R.item())
                    setattr(self, f'QS{n}', QS.item())
                    setattr(self, f'exp_reward{n}', exp_reward.item())

                    if f'R{n}' in self.history.keys():
                        self.history[f'R{n}'].append(R.item())
                    else:
                        self.history[f'R{n}'] = []
                    if f'QS{n}' in self.history.keys():
                        self.history[f'QS{n}'].append(QS.item())
                    else:
                        self.history[f'QS{n}'] = []
                    if f'exp_reward{n}' in self.history.keys():
                        self.history[f'exp_reward{n}'].append(exp_reward.item())
                    else:
                        self.history[f'exp_reward{n}'] = []

                setattr(self, 'exp_reward', cum_exp)
                self.history[f'exp_reward'].append(cum_exp)
                self.history['num_update'] = self.update_count
        if self.use_threshold and self.exp_reward >= self.threshold:
            self._train = False
        return True
