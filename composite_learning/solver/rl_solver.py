import datetime
import time
import os
from dataclasses import dataclass
import logging
from typing import List
from typing import Callable
from typing import Union

import torch
import torch.nn as nn

from solver import base
import network
import data


@dataclass(kw_only=True)
class RLPerceptronSolver(base.BaseSolver):
    teacher_network: network.BaseTeacher
    student_network: network.BaseStudent
    lr: float
    weight_decay: float
    optimizer_type: str

    def __post_init__(self):
        self._train = False

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

    def train(self, data_loader, num_iter):
        self._setup_train()
        for i in range(num_iter):
            x = data_loader.get_batch()['x']
            loss = self._step(x)
            if loss is None:
                print('invalid batch')
        self._train = False

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
