import abc
import os
import logging
import math
from typing import List
from typing import Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np

import network
import data


@dataclass
class TwoPhaseBaseSolver(abc.ABC):
    teachers: List[network.BaseTeacher]
    students: List[network.BaseStudent]
    dataloaders: List[data.BaseDataLoader]
    logdir: Union[str, None]

    def __post_init__(self):
        assert len(self.teachers) == len(
            self.students), "Number of students and teachers should be the same"
        if not self.logdir is None:
            self._setup_logger()
        self.seq_len = self.dataloaders[0].seq_len[0]
        self.num_base_tasks = len(self.teachers)
        self.network_size = self.teachers[0].input_dimension
        self.Q = np.zeros(shape=(self.num_base_tasks, self.num_base_tasks))
        self.R = np.zeros(shape=(self.num_base_tasks, self.num_base_tasks))
        self.S = np.zeros(shape=(self.num_base_tasks, self.num_base_tasks))

    def train(self, nums_iter: List[int], lrs: List[tuple],
              update_frequency: int):
        self._setup_history()
        self.update_frequency = update_frequency
        self.lr1_w = lrs[0][0]
        self.lr1_v = lrs[0][1]
        self.lr2_w = lrs[1][0]
        self.lr2_v = lrs[1][1]

        self.phase1 = True
        self.phase2 = False

        for n in range(nums_iter[0]):
            x = self.dataloaders[0].get_batch()['x']
            self._phase1_step(x)
            if n % self.update_frequency == 0:
                self.metric()

        self.phase1 = False
        self.phase2 = True
        for n in range(nums_iter[1]):
            x = self.dataloaders[1].get_batch()['x']
            self._phase2_step(x)
            if n % self.update_frequency == 0:
                self.metric()

    @abc.abstractmethod
    def _phase1_step(self, x):
        return NotImplementedError

    @abc.abstractmethod
    def _phase2_step(self, x):
        return NotImplementedError

    def _setup_logger(self):
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        logging.basicConfig(filename=os.path.join(self.logdir, f"log.log"),
                            format='%(asctime)s %(message)s',
                            filemode='w')
        self.logger = logging.getLogger()

        handler = logging.FileHandler(
            filename=os.path.join(self.logdir, f"log.log"))
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def _setup_history(self):
        self.history = {
            'phase1': {
                'P': [],
                'R': [],
                'Q': [],
                'P_tilde': [],
                'empirical_P_tilde': [],
                'VS': []
            },
            'phase2': {
                'R': [],
                'Q': [],
                'P': [],
                'P_tilde': [],
                'empirical_P_tilde': [],
                'VS': []
            }
        }

    def _history_update(self, **kwargs):
        if self.phase1 and not self.phase2:
            for k, v in kwargs.items():
                self.history['phase1'][k].append(v)

        if not self.phase1 and self.phase2:
            for k, v in kwargs.items():
                self.history['phase2'][k].append(v)


@dataclass(kw_only=True)
class TwoPhaseContextSolver(TwoPhaseBaseSolver):
    context_teacher: network.BaseTeacher
    context_student: network.BaseStudent
    identical: bool

    def _phase1_inference(self, x):
        ys_hat = []
        ys_sign_hat = []
        ys = []
        ys_sign = []
        for i in range(self.num_base_tasks):
            y_hat, y_sign_hat = self.students[i](x[i])
            y, y_sign = self.teachers[i](x[i])
            ys_hat.append(y_hat)
            ys_sign_hat.append(y_sign_hat)
            ys.append(y)
            ys_sign.append(y_sign)
        ys_hat = torch.stack(ys_hat, dim=-1)
        ys_sign_hat = torch.stack(ys_sign_hat, dim=-1)
        ys = torch.stack(ys, dim=-1)
        ys_sign = torch.stack(ys_sign, dim=-1)

        return ys_hat.squeeze(), ys_sign_hat.squeeze(), ys.squeeze(
        ), ys_sign.squeeze()

    def _phase2_inference(self, x):
        ys_tilde_hat = []
        ys_tilde_sign_hat = []
        ys_tilde = []
        ys_tilde_sign = []
        ys_hat, _, ys, _ = self._phase1_inference(x)
        for i in range(self.seq_len):
            y_tilde, y_tilde_sign = self.context_teacher(ys[i][None, :])
            y_tilde_hat, y_tilde_sign_hat = self.context_student(
                ys_hat[i][None, :])
            ys_tilde.append(y_tilde.item())
            ys_tilde_sign.append(y_tilde_sign.item())
            ys_tilde_hat.append(y_tilde_hat.item())
            ys_tilde_sign_hat.append(y_tilde_sign_hat.item())

        return torch.tensor(ys_tilde_hat), torch.tensor(
            ys_tilde_sign_hat), torch.tensor(ys_tilde), torch.tensor(
                ys_tilde_sign)

    def _phase1_step(self, x):
        dw = []
        for i in range(self.num_base_tasks):
            y_hat, y_sign_hat = self.students[i](x[i])
            y, y_sign = self.teachers[i](x[i])
            if not torch.all(y_sign_hat == y_sign):
                dw.append(None)
            else:
                dw.append(self.lr1_w / math.sqrt(self.network_size) *
                          y_sign_hat * x[i])
        for i in range(self.num_base_tasks):
            if dw[i] is not None:
                self.students[i].layers[0].weight.data += dw[i].squeeze(
                    dim=0).mean(dim=0)

    def _phase2_step(self, x):
        ys_tilde_sign_hat = []
        ys_tilde_sign = []
        ys_hat, _, ys, _ = self._phase1_inference(x)
        for i in range(self.seq_len):
            _, y_tilde_sign = self.context_teacher(ys[i][None, :])
            _, y_tilde_sign_hat = self.context_student(ys_hat[i][None, :])
            if y_tilde_sign != y_tilde_sign_hat:
                return None

            ys_tilde_sign.append(y_tilde_sign.item())
            ys_tilde_sign_hat.append(y_tilde_sign_hat.item())
        for i in range(self.num_base_tasks):
            dw = self.lr2_w / math.sqrt(self.network_size) * torch.tensor(
                ys_tilde_sign_hat) * self.context_student.layers[0].weight.data[
                    0, i] * x[i].squeeze().T
            self.students[i].layers[0].weight.data += dw.mean(dim=1)
            dv = torch.mean(self.lr2_v / self.network_size * ys_hat[:, i] *
                            torch.tensor(ys_tilde_sign_hat))
            self.context_student.layers[0].weight.data[0, i] += dv

    def metric(self):
        task_p = []
        for (t, s) in zip(self.teachers, self.students):
            r = s.layers[0].weight.data @ t.layers[
                0].weight.data.T / self.network_size
            q = s.layers[0].weight.data @ s.layers[
                0].weight.data.T / self.network_size
            overlap = r.item() / math.sqrt(q.item())
            p = 1 - 1 / math.pi * math.acos(overlap)
            task_p.append(p)
        self._history_update(P=task_p)

        for i in range(self.num_base_tasks):
            for j in range(self.num_base_tasks):
                self.Q[i][j] = (
                    1 / self.network_size *
                    self.students[i].layers[0].weight.data
                    @ self.students[j].layers[0].weight.data.T).item()
                self.R[i][j] = (
                    1 / self.network_size *
                    self.students[i].layers[0].weight.data
                    @ self.teachers[j].layers[0].weight.data.T).item()
                self.S[i][j] = (
                    1 / self.network_size *
                    self.teachers[i].layers[0].weight.data
                    @ self.teachers[j].layers[0].weight.data.T).item()

        R = self.R.copy()
        Q = self.Q.copy()
        VS = self.context_student.layers[0].weight.data[0].numpy().copy()
        self._history_update(R=R, Q=Q)
        self._history_update(VS=VS)
        if not self.identical:
            norm_student = np.sqrt(
                np.sum([
                    self.context_student.layers[0].weight.data[0, i] *
                    self.context_student.layers[0].weight.data[0, i] *
                    self.Q[i][i] for i in range(self.num_base_tasks)
                ]))
            norm_teacher = np.sqrt(
                np.sum([
                    self.context_teacher.layers[0].weight.data[0, i] *
                    self.context_teacher.layers[0].weight.data[0, i] *
                    self.S[i][i] for i in range(self.num_base_tasks)
                ]))

            angle = np.arccos(
                sum([
                    self.context_student.layers[0].weight.data[0, i] *
                    self.context_teacher.layers[0].weight.data[0, i] *
                    self.R[i][i] for i in range(self.num_base_tasks)
                ]) / norm_teacher / norm_student)
        else:
            norm_student = np.sqrt(
                np.sum([
                    self.context_student.layers[0].weight.data[0, i] *
                    self.context_student.layers[0].weight.data[0, j] *
                    self.Q[i][j]
                    for i in range(self.num_base_tasks)
                    for j in range(self.num_base_tasks)
                ]))
            norm_teacher = np.sqrt(
                np.sum([
                    self.context_teacher.layers[0].weight.data[0, i] *
                    self.context_teacher.layers[0].weight.data[0, j] *
                    self.S[i][j]
                    for i in range(self.num_base_tasks)
                    for j in range(self.num_base_tasks)
                ]))

            angle = np.arccos(
                np.sum([
                    self.context_student.layers[0].weight.data[0, i] *
                    self.context_teacher.layers[0].weight.data[0, j] *
                    self.R[i][j]
                    for i in range(self.num_base_tasks)
                    for j in range(self.num_base_tasks)
                ]) / norm_teacher / norm_student)

        P = 1 - angle / np.pi
        self._history_update(P_tilde=P)

        samples = []
        for _ in range(100):
            x = self.dataloaders[2].get_batch()['x']
            _, ys_tilde_sign_hat, _, ys_tilde_sign = self._phase2_inference(x)

            samples.append(
                np.mean([ys_tilde_sign.numpy() == ys_tilde_sign_hat.numpy()]))

        empirical_P_tilde = sum(samples) / len(samples)

        self._history_update(empirical_P_tilde=empirical_P_tilde)


@dataclass(kw_only=True)
class TwoPhaseContextTSSolver(TwoPhaseContextSolver):
    context_teachers: network.BaseTeacher
    context_students: network.BaseStudent

    def __post_init__(self):
        assert len(self.teachers) == len(
            self.students), "Number of students and teachers should be the same"
        if not self.logdir is None:
            self._setup_logger()
        self.seq_len = self.dataloaders[0].seq_len[0]
        self.num_base_tasks = len(self.teachers)
        self.network_size = self.teachers[0].input_dimension
        self.context_network_size = self.context_teachers[0].input_dimension
        self.count = 0
        self.Q = np.zeros(shape=(self.num_base_tasks, self.num_base_tasks))
        self.R = np.zeros(shape=(self.num_base_tasks, self.num_base_tasks))
        self.S = np.zeros(shape=(self.num_base_tasks, self.num_base_tasks))
        self.QV = np.zeros(shape=(self.num_base_tasks, self.num_base_tasks))
        self.RV = np.zeros(shape=(self.num_base_tasks, self.num_base_tasks))
        self.SV = np.zeros(shape=(self.num_base_tasks, self.num_base_tasks))

    def _phase2_inference(self, x1, x2):
        ys_tilde_hat = []
        ys_tilde_sign_hat = []
        ys_tilde = []
        ys_tilde_sign = []

        cs_hat = []
        cs = []
        ys_hat, _, ys, _ = self._phase1_inference(x1)
        for i in range(self.num_base_tasks):
            c, _ = self.context_teachers[i](x2[i])
            c_hat, _ = self.context_students[i](x2[i])
            cs.append(c)
            cs_hat.append(c_hat)
        cs = torch.vstack(cs).squeeze().T
        cs_hat = torch.vstack(cs_hat).squeeze().T
        for i in range(self.seq_len):
            y_tilde = cs[i] @ ys[i]
            y_tilde_sign = torch.sign(y_tilde)
            y_tilde_hat = cs_hat[i] @ ys_hat[i]
            y_tilde_sign_hat = torch.sign(y_tilde_hat)
            ys_tilde_sign.append(y_tilde_sign.item())
            ys_tilde_sign_hat.append(y_tilde_sign_hat.item())
            ys_tilde.append(y_tilde.item())
            ys_tilde_hat.append(y_tilde_hat.item())

        return torch.tensor(ys_tilde_hat), torch.tensor(
            ys_tilde_sign_hat), torch.tensor(ys_tilde), torch.tensor(
                ys_tilde_sign)

    def _phase2_step(self, x1, x2):

        ys_tilde_hat = []
        ys_tilde_sign_hat = []
        ys_tilde = []
        ys_tilde_sign = []

        cs_hat = []
        cs = []
        ys_hat, _, ys, _ = self._phase1_inference(x1)
        for i in range(self.num_base_tasks):
            c, _ = self.context_teachers[i](x2[i])
            c_hat, _ = self.context_students[i](x2[i])
            cs.append(c)
            cs_hat.append(c_hat)
        cs = torch.vstack(cs).squeeze().T
        cs_hat = torch.vstack(cs_hat).squeeze().T
        for i in range(self.seq_len):
            y_tilde = cs[i] @ ys[i]
            y_tilde_sign = torch.sign(y_tilde)
            y_tilde_hat = cs_hat[i] @ ys_hat[i]
            y_tilde_sign_hat = torch.sign(y_tilde_hat)
            if y_tilde_sign != y_tilde_sign_hat:
                self.history['phase2']['counts'].append(self.count)
                return None
            ys_tilde_sign.append(y_tilde_sign)
            ys_tilde_sign_hat.append(y_tilde_sign_hat)

        for i in range(self.num_base_tasks):
            dw = self.lr2_w / math.sqrt(self.network_size) * torch.tensor(
                ys_tilde_sign_hat) * cs_hat[:, i] * x1[i].squeeze().T
            dv = self.lr2_v / math.sqrt(
                self.context_network_size) * torch.tensor(
                    ys_tilde_sign_hat) * ys_hat[:, i] * x2[i].squeeze().T

            self.students[i].layers[0].weight.data += dw.mean(dim=1)
            self.context_students[i].layers[0].weight.data += dv.mean(dim=1)
        self.count += 1
        self.history['phase2']['counts'].append(self.count)

    def metric(self):
        reward_rate = 1
        for k, (t, s) in enumerate(zip(self.teachers, self.students)):
            r = t.layers[0].weight.data @ s.layers[
                0].weight.data.T / self.network_size
            q = s.layers[0].weight.data @ s.layers[
                0].weight.data.T / self.network_size
            overlap = r.item() / math.sqrt(q.item())
            p = 1 - 1 / math.pi * math.acos(overlap)
            reward_rate *= (p**self.seq_len)
        self._history_update(reward_rate=reward_rate)

        for i in range(self.num_base_tasks):
            for j in range(self.num_base_tasks):
                self.Q[i][j] = (
                    1 / self.network_size *
                    self.students[i].layers[0].weight.data
                    @ self.students[j].layers[0].weight.data.T).item()
                self.R[i][j] = (
                    1 / self.network_size *
                    self.students[i].layers[0].weight.data
                    @ self.teachers[j].layers[0].weight.data.T).item()
                self.S[i][j] = (
                    1 / self.network_size *
                    self.teachers[i].layers[0].weight.data
                    @ self.teachers[j].layers[0].weight.data.T).item()

        R = self.R.copy()
        Q = self.Q.copy()

        self._history_update(R=R, Q=Q)

        if True:

            for i in range(self.num_base_tasks):
                for j in range(self.num_base_tasks):
                    self.QV[i][j] = (
                        1 / self.context_network_size *
                        self.context_students[i].layers[0].weight.data
                        @ self.context_students[j].layers[0].weight.data.T
                    ).item()
                    self.RV[i][j] = (
                        1 / self.context_network_size *
                        self.context_students[i].layers[0].weight.data
                        @ self.context_teachers[j].layers[0].weight.data.T
                    ).item()
                    self.SV[i][j] = (
                        1 / self.context_network_size *
                        self.context_teachers[i].layers[0].weight.data
                        @ self.context_teachers[j].layers[0].weight.data.T
                    ).item()
            RV = self.RV.copy()
            QV = self.QV.copy()
            self._history_update(RV=RV, QV=QV)
            samples = []

            for _ in range(100):
                x1 = self.dataloaders[0].get_batch()['x']
                x2 = self.dataloaders[1].get_batch()['x']
                _, ys_tilde_sign_hat, _, ys_tilde_sign = self._phase2_inference(
                    x1, x2)

                samples.append(
                    np.mean(
                        [ys_tilde_sign.numpy() == ys_tilde_sign_hat.numpy()]))

            empirical_overlap = sum(samples) / len(samples)

            self._history_update(overlap=empirical_overlap)

    def _setup_history(self):
        self.history = {
            'phase1': {
                'overlap': [],
                'context_similarity': [],
                'reward_rate': [],
                'R': [],
                'Q': [],
                'overlap': [],
                'counts': [],
                'RV': [],
                'QV': []
            },
            'phase2': {
                'overlap': [],
                'context_similarity': [],
                'reward_rate': [],
                'R': [],
                'Q': [],
                'RV': [],
                'QV': [],
                'P': [],
                'counts': [],
                'norm_student': []
            }
        }

    def train(self, nums_iter: List[int], lrs: List[tuple],
              update_frequency: int):
        self._setup_history()
        self.update_frequency = update_frequency
        self.lr1_w = lrs[0][0]
        self.lr1_v = lrs[0][1]
        self.lr2_w = lrs[1][0]
        self.lr2_v = lrs[1][1]
        self.dw = []
        self.dv = []

        self.phase1 = True
        self.phase2 = False

        for n in range(nums_iter[0]):
            x = self.dataloaders[0].get_batch()['x']
            self._phase1_step(x)
            if n % self.update_frequency == 0:
                self.metric()

        self.phase1 = False
        self.phase2 = True
        for n in range(nums_iter[1]):
            x1 = self.dataloaders[0].get_batch()['x']
            x2 = self.dataloaders[1].get_batch()['x']
            self._phase2_step(x1, x2)
            if n % self.update_frequency == 0:
                self.metric()
