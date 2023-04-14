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
class TeacherStudentBaseSolver(base.BaseSolver):
    teacher_network: Union[network.BaseTeacher, network.BaseEnsemble]
    student_network: Union[network.BaseStudent, network.BaseEnsemble]
    weight_decay: float

    def __post_init__(self):
        self._rejection_func = None
        self._train = False

    def _setup_history(self):
        self.history = {'loss': []}

    def _setup_train(self):
        self._train = True
        self._setup_history()
        if self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                params=self.student_network.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay)

    def _history_update(self, **kwargs):
        for k, v in kwargs.items():
            self.history[k].append(v)

    def _compose_teacher_y(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplementedError

    def _compose_student_y(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplementedError

    def _step(self, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        x = self._get_valid_batch(x, self._rejection_func)
        if x is None:
            return None
        teacher_output, student_output = self.inference(x, self._rejection_func)
        teacher_y = self._compose_teacher_y(teacher_output)
        student_y = self._compose_student_y(student_output)
        loss = self.criterion(teacher_y, student_y)
        loss.sum().backward()
        self.optimizer.step()
        loss = loss.sum().item()
        self._history_update(loss=loss,
                             prediction=student_y.detach(),
                             true=teacher_y.detach())

        return loss

    def _get_valid_batch(self, x: torch.Tensor,
                         rejection_func: Union[Callable, None]) -> torch.Tensor:
        with torch.no_grad():
            teacher_output = self.teacher_network.forward(x)
            if rejection_func is not None:
                x = rejection_func(x, teacher_output)
                if x is None:
                    return None
                return x
        return x

    def inference(self, x: torch.Tensor,
                  rejection_func: Union[Callable, None]) -> torch.Tensor:
        x = self._get_valid_batch(x, rejection_func)
        if x is None:
            return None
        if not self._train:
            with torch.no_grad():
                return self.teacher_network.forward(
                    x), self.student_network.forward(x)
        else:
            return self.teacher_network.forward(
                x), self.student_network.forward(x)

    def train(self, data_loader: data.BaseDataLoader, num_iter: int,
              rejection_func: Callable) -> torch.Tensor:
        """TODO
        Better way to add rejection method; 
        Currently we get function as an rejection_func argument to reject invalid x depends on the training tasks. 
        rejection_func: x -> corrected_x, corrected_teacher_outputs
        """
        self._setup_train()
        self._rejection_func = rejection_func
        #self._setup_logger()

        for i in range(num_iter):
            x = data_loader.get_batch()['x']
            loss = self._step(x)
            if loss is None:
                self._history_update(loss='invalid_batch',
                                     prediction='invalid_batch',
                                     true='invalid_batch')
        self._train = False
        #self.logger.info(f'Iter: {i}\tLoss: {loss:.5f}')


@dataclass
class SingleTeacherStudentRegressionSolver(TeacherStudentBaseSolver):

    def _setup_history(self):
        self.history = {'loss': [], 'prediction': [], 'true': []}

    def _compose_teacher_y(self, x):
        return torch.multiply(*x)

    def _compose_student_y(self, x):
        return x


@dataclass
class EnsembleTeacherStudentRegressionSolver(TeacherStudentBaseSolver):

    def _setup_history(self):
        self.history = {'loss': [], 'prediction': [], 'true': []}

    def _compose_teacher_y(self, x):
        return torch.add(
            *[torch.multiply(*network_output) for network_output in x])

    def _compose_student_y(self, x):
        return x


@dataclass
class EnsemblePartitionedTeacherStudentRegressionSolver(TeacherStudentBaseSolver
                                                       ):

    def _setup_history(self):
        self.history = {
            'loss': [],
            'prediction': [],
            'true': [],
            'y1': [],
            'y2': [],
            'y3': [],
            'y4': []
        }

    def _compose_teacher_y(self, x):
        return torch.add(*[
            torch.multiply(x[i * 2], x[i * 2 + 1])
            for i in range(int(len(x) // 2))
        ])

    def _compose_student_y(self, x):
        return x


@dataclass
class TeacherStudentElementwiseRegressionSolver(TeacherStudentBaseSolver):

    def _setup_history(self):
        self.history = {'loss': [], 'prediction': [], 'true': []}

    def _compose_teacher_y(self, x):
        return torch.stack(x, dim=1).squeeze()

    def _compose_student_y(self, x):
        return torch.stack(x, dim=1).squeeze()