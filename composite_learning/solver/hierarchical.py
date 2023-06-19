import abc
import os
import logging
import math
from typing import List
from typing import Union
from dataclasses import dataclass
import torch
import torch.nn as nn

import network
import data


@dataclass
class TwoPhaseBaseSolver(abc.ABC):
    teachers: List[network.BaseTeacher]
    students: List[network.BaseStudent]
    dataloader: data.BaseDataLoader
    logdir: Union[str, None]

    def __post_init__(self):
        assert len(self.teachers) == len(
            self.students), "Number of students and teachers should be the same"
        if not self.logdir is None:
            self._setup_logger()
        self.seq_len = self.dataloader.seq_len[0]
        self.num_base_tasks = len(self.teachers)
        self.network_size = self.teachers[0].input_dimension

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
            x = self.dataloader.get_batch()['x']
            self._phase1_step(x)
            if n % self.update_frequency == 0:
                self.metric()

        self.phase1 = False
        self.phase2 = True
        for n in range(nums_iter[1]):
            x = self.dataloader.get_batch()['x']
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
                'overlap': [],
                'context_similarity': [],
                'reward_rate': []
            },
            'phase2': {
                'overlap': [],
                'context_similarity': [],
                'reward_rate': []
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
                return None
            dw.append(self.lr1_w / math.sqrt(self.network_size) * y_sign_hat *
                      x[i])
        for i in range(self.num_base_tasks):
            self.students[i].layers[0].weight.data += dw[i].squeeze(dim=0).mean(
                dim=0)

    def _phase2_step(self, x):
        dw = []
        dv = []
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
                ys_tilde_sign) * self.context_student.layers[0].weight.data[
                    0, i] * x[i].squeeze().T
            self.students[i].layers[0].weight.data += dw.mean(dim=1)
            dv = torch.mean(self.lr2_v / math.sqrt(self.network_size) *
                            ys[:, i] * torch.tensor(ys_tilde_sign_hat))
            self.context_student.layers[0].weight.data[0, i] += dv

    def metric(self):
        reward_rate = 1
        for k, (t, s) in enumerate(zip(self.teachers, self.students)):
            r = t.layers[0].weight.data @ s.layers[0].weight.data.T / 100
            q = s.layers[0].weight.data @ s.layers[0].weight.data.T / 100
            overlap = r.item() / math.sqrt(q.item())
            p = 1 - 1 / math.pi * math.acos(overlap)
            reward_rate *= p**self.seq_len
        self._history_update(reward_rate=reward_rate)
        if self.phase2:
            tc_norm = self.context_teacher.layers[
                0].weight.data @ self.context_teacher.layers[0].weight.data.T
            sc_norm = self.context_student.layers[
                0].weight.data @ self.context_student.layers[0].weight.data.T
            tc_sc = self.context_teacher.layers[
                0].weight.data @ self.context_student.layers[0].weight.data.T
            context_similarity = (tc_sc / torch.sqrt(tc_norm) /
                                  torch.sqrt(sc_norm)).item()

            samples = []
            for _ in range(100):
                x = self.dataloader.get_batch()['x']
                _, ys_tilde_sign_hat, _, ys_tilde_sign = self._phase2_inference(
                    x)

                samples.append(
                    (ys_tilde_sign * ys_tilde_sign_hat + 1).mean().item())

            empirical_overlap = sum(samples) / len(samples)

            self._history_update(context_similarity=context_similarity,
                                 overlap=empirical_overlap)
