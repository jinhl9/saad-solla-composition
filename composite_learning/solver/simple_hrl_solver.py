from dataclasses import dataclass
from typing import List
from typing import Union
import numpy as np
import data
import solver.ode as ode

sqrt2pi = np.sqrt(2 * np.pi)


@dataclass
class BaseSimulator():
    input_dim: int
    seq_len: int

    def __post_init__(self):
        pass

    def setup_train(self):
        raise NotImplementedError

    def setup_history(self, num_update):
        self.history = {}

    def update_history(self, **kwargs):
        raise NotImplementedError

    def step(self, lr):
        raise NotImplementedError

    def inference(self, x):
        raise NotImplementedError

    def train(self, num_iter, update_frequency, lr):
        self.lr = lr
        self.setup_history(num_update=num_iter // update_frequency)
        self.setup_train()
        for i in range(num_iter):
            self.step()
            if i % update_frequency == 0:
                self.update_history(history_index=i // update_frequency)

    def __str__(self):
        raise NotImplementedError


@dataclass
class MultipleRLPerceptronSimulator(BaseSimulator):
    num_task: int
    identical: bool

    def setup_train(self):
        self.lr_w = self.lr['lr_w']
        self.WT = np.random.normal(loc=0.0,
                                   scale=1.0,
                                   size=(self.num_task, self.input_dim))
        self.WT /= np.sqrt((np.sum(self.WT @ self.WT.T)) / self.input_dim)
        self.WS = np.random.normal(loc=0.0,
                                   scale=1.0,
                                   size=(self.num_task, self.input_dim))

    def setup_history(self, num_update):
        self.history = {
            'Q': np.zeros((num_update, self.num_task, self.num_task)),
            'R': np.zeros((num_update, self.num_task, self.num_task)),
            'P': np.zeros((num_update, self.num_task))
        }

    def update_history(self, history_index):
        Q = self.WS @ self.WS.T / self.input_dim
        R = self.WS @ self.WT.T / self.input_dim
        P = 1 - np.arccos(np.diag(R) / np.sqrt(np.diag(Q))) / np.pi

        self.history['Q'][history_index] = Q
        self.history['R'][history_index] = R
        self.history['P'][history_index] = P

    def step(self):
        if self.identical:
            x = np.random.normal(loc=0.0,
                                 scale=1.0,
                                 size=(1, self.input_dim, self.seq_len))
            x = np.repeat(x, self.num_task, axis=0)
        else:
            x = np.random.normal(loc=0.0,
                                 scale=1.0,
                                 size=(self.num_task, self.input_dim,
                                       self.seq_len))

        y, y_sign, y_hat, y_hat_sign = self.inference(x)

        for k in range(self.num_task):
            if (y_sign[k] == y_hat_sign[k]).all():
                dW = (1 / np.sqrt(self.input_dim) * y_sign[k] *
                      x[k]).mean(axis=-1)
                self.WS[k] += self.lr_w * dW
                self.WS[k] = np.divide(self.WS[k] * np.sqrt(self.input_dim),
                                       np.linalg.norm(self.WS[k]))

    def inference(self, x):
        y = np.diagonal(self.WT @ x).T / np.sqrt(self.input_dim)
        y_hat = np.diagonal(self.WS @ x).T / np.sqrt(self.input_dim)
        y_sign = np.sign(y)
        y_hat_sign = np.sign(y_hat)

        return y, y_sign, y_hat, y_hat_sign


@dataclass
class CompositionalTaskSimulator(BaseSimulator):
    num_task: int
    identical: bool
    WT: Union[None, np.array] = None
    WS: Union[None, np.array] = None
    VT: Union[None, np.array] = None
    VS: Union[None, np.array] = None
    V_norm: int = 0

    def setup_train(self):
        self.lr_wc = self.lr['lr_wc']
        self.lr_vc = self.lr['lr_vc']
        if self.WT is None:
            self.WT = np.random.normal(loc=0.0,
                                       scale=1.0,
                                       size=(self.num_task, self.input_dim))
        for w in self.WT:
            w /= np.sqrt(w @ w.T / self.input_dim)
        if self.WS is None:
            self.WS = np.random.normal(loc=0.0,
                                       scale=1.0,
                                       size=(self.num_task, self.input_dim))
        if self.VT is None:
            self.VT = np.random.normal(loc=0.0, scale=1.0, size=(self.num_task))
        if self.VS is None:
            self.VS = np.random.normal(loc=0.0, scale=1.0, size=(self.num_task))

        self.S = self.WT @ self.WT.T / self.input_dim

    def setup_history(self, num_update):
        self.history = {
            'Q': np.zeros((num_update, self.num_task, self.num_task)),
            'R': np.zeros((num_update, self.num_task, self.num_task)),
            'P': np.zeros((num_update, self.num_task)),
            'overlap': np.zeros((num_update, self.num_task)),
            'VS': np.zeros((num_update, self.num_task)),
            'VSVT': np.zeros((num_update, self.num_task)),
            'P_tilde': np.zeros((num_update)),
            'overlap_tilde': np.zeros((num_update))
        }

    @property
    def norm_student(self):
        if self.identical:
            return np.sqrt(
                np.sum([
                    self.VS[i] * self.VS[j] * self.Q[i][j]
                    for i in range(self.num_task)
                    for j in range(self.num_task)
                ]))

        else:
            return np.sqrt(
                np.sum([
                    self.VS[i] * self.VS[i] * self.Q[i][i]
                    for i in range(self.num_task)
                ]))

    @property
    def norm_teacher(self):
        if self.identical:
            return np.sqrt(
                np.sum([
                    self.VT[i] * self.VT[j] * self.S[i][j]
                    for i in range(self.num_task)
                    for j in range(self.num_task)
                ]))
        else:
            return np.sqrt(
                np.sum([
                    self.VT[i] * self.VT[i] * self.S[i][i]
                    for i in range(self.num_task)
                ]))

    @property
    def Q(self):
        return self.WS @ self.WS.T / self.input_dim

    @property
    def R(self):
        return self.WS @ self.WT.T / self.input_dim

    @property
    def overlap_task(self):

        return np.diagonal(self.R) / np.sqrt(np.diagonal(self.Q))

    @property
    def P_task(self):
        return 1 - np.arccos(self.overlap_task) / np.pi

    @property
    def overlap(self):
        if self.identical:
            return np.sum([
                self.VS[i] * self.VT[j] * self.R[i][j]
                for i in range(self.num_task)
                for j in range(self.num_task)
            ]) / self.norm_teacher / self.norm_student

        if not self.identical:
            return np.sum([
                self.VS[i] * self.VT[i] * self.R[i][i]
                for i in range(self.num_task)
            ]) / self.norm_teacher / self.norm_student

    @property
    def P_tilde(self):
        return 1 - np.arccos(self.overlap) / np.pi

    def update_history(self, history_index):

        self.history['Q'][history_index] = self.Q
        self.history['R'][history_index] = self.R
        self.history['P'][history_index] = self.P_task
        self.history['overlap'][history_index] = self.overlap_task
        self.history['VS'][history_index] = self.VS
        self.history['P_tilde'][history_index] = self.P_tilde
        self.history['overlap_tilde'][history_index] = self.overlap
        self.history['VSVT'][history_index] = np.dot(
            self.VS, self.VT) / np.linalg.norm(self.VS) / np.linalg.norm(
                self.VT)

    def step(self):

        if self.identical:
            x = np.random.normal(loc=0.0,
                                 scale=1.0,
                                 size=(1, self.input_dim, self.seq_len))
            x = np.repeat(x, self.num_task, axis=0)
        else:
            x = np.random.normal(loc=0.0,
                                 scale=1.0,
                                 size=(self.num_task, self.input_dim,
                                       self.seq_len))

        (y, y_sign, y_hat,
         y_hat_sign), (y_tilde, y_tilde_hat, y_tilde_sign,
                       y_tilde_hat_sign) = self.inference(x)  #num_task*n_seq
        if (y_tilde_sign == y_tilde_hat_sign).all():
            dW = (1 / np.sqrt(self.input_dim) * y_tilde_hat_sign[:, None].T *
                  self.VS[:, None] * x.swapaxes(0, 1)).mean(axis=-1).T
            dV = (1 / self.input_dim * y_hat * y_tilde_hat_sign).mean(axis=-1)
            self.WS += self.lr_wc * dW
            self.VS += self.lr_vc * dV
            if self.V_norm != 0:
                self.VS *= np.sqrt(self.num_task) / np.linalg.norm(self.VS)
            else:
                self.VS /= np.linalg.norm(self.VS)

            self.WS = np.divide(self.WS * np.sqrt(self.input_dim),
                                np.linalg.norm(self.WS, axis=1)[:, None])

    def inference(self, x):

        def single_task_inference(x):
            y = np.diagonal(self.WT @ x).T / np.sqrt(self.input_dim)
            y_hat = np.diagonal(self.WS @ x).T / np.sqrt(self.input_dim)
            y_sign = np.sign(y)
            y_hat_sign = np.sign(y_hat)

            return y, y_sign, y_hat, y_hat_sign

        y, y_sign, y_hat, y_hat_sign = single_task_inference(x)

        y_tilde = self.VT @ y
        y_tilde_hat = self.VS @ y_hat
        y_tilde_sign = np.sign(y_tilde)
        y_tilde_hat_sign = np.sign(y_tilde_hat)

        return (y, y_sign, y_hat, y_hat_sign), (y_tilde, y_tilde_hat,
                                                y_tilde_sign, y_tilde_hat_sign)

@dataclass
class CurriculumCompositionalTaskSimulator(CompositionalTaskSimulator):
    num_task: int
    identical: bool
    WT: Union[None, np.array] = None
    WS: Union[None, np.array] = None
    VT: Union[None, np.array] = None
    VS: Union[None, np.array] = None
    V_norm: int = 0

    def __post_init__(self):
        self.multipleRLPerceptron = MultipleRLPerceptronSimulator(
            self.input_dim, self.seq_len, self.num_task, self.identical)

    def pretrain_step(self):
        return self.multipleRLPerceptron.step()

    def train(self, num_iter, update_frequency, lr):
        self.lr = lr
        self.setup_history(num_update=sum(num_iter) // update_frequency)
        self.setup_train()

        self.multipleRLPerceptron.WS = self.WS
        self.multipleRLPerceptron.WT = self.WT
        self.multipleRLPerceptron.lr_w = self.lr['lr_w']

        for i in range(num_iter[0]):
            self.pretrain_step()
            if i % update_frequency == 0:
                self.update_history(history_index=i // update_frequency)

        for i in range(num_iter[1]):
            self.step()
            if i % update_frequency == 0:
                self.update_history(history_index=(num_iter[0] + i) //
                                    update_frequency)


class HRLODESolver(ode.ODESolver):

    def __init__(self, VS: np.array, VT: np.array, WS: np.array, WT: np.array,
                 lr_ws: List[float], lr_v: float, seq_length: int, N: int,
                 V_norm: int):
        super().__init__()
        self.num_task = len(VS)
        self.seq_length = seq_length
        self.VS = VS
        self.VT = VT
        self.N = N
        self.lr_w1 = lr_ws[0]
        self.lr_w2 = lr_ws[1]
        self.lr_v = lr_v
        self.WS = WS
        self.WT = WT
        self.V_norm = V_norm

        for w in self.WT:
            w /= np.sqrt(w @ w.T / self.N)

    def _setup_history(self, num_update):
        self.history = {
            'phase1': {
                'Q': np.zeros((num_update[0], self.num_task, self.num_task)),
                'R': np.zeros((num_update[0], self.num_task, self.num_task)),
                'P': np.zeros((num_update[0], self.num_task)),
                'overlap': np.zeros((num_update[0], self.num_task)),
                'P_tilde': np.zeros((num_update[0])),
                'overlap_tilde': np.zeros((num_update[0])),
                'VS': np.zeros((num_update[0], self.num_task)),
                'VT': self.VT
            },
            'phase2': {
                'Q': np.zeros((num_update[1], self.num_task, self.num_task)),
                'R': np.zeros((num_update[1], self.num_task, self.num_task)),
                'P': np.zeros((num_update[1], self.num_task)),
                'overlap': np.zeros((num_update[1], self.num_task)),
                'P_tilde': np.zeros((num_update[1])),
                'overlap_tilde': np.zeros((num_update[1])),
                'VS': np.zeros((num_update[1], self.num_task)),
                'VT': self.VT,
                'VSVT': np.zeros((num_update[1]))
            }
        }

    def _init_params(self):
        if self.WS is not None and self.WT is not None:

            self.Q = self.WS @ self.WS.T / self.N
            self.R = self.WS @ self.WT.T / self.N
            self.S = self.WT @ self.WT.T / self.N

        else:
            for i in range(self.num_task):
                self.Q = np.zeros(shape=(self.num_task, self.num_task))
                self.R = np.zeros(shape=(self.num_task, self.num_task))
                self.S = np.zeros(shape=(self.num_task, self.num_task))
                self.Q[i][i] += 1.
                self.S[i][i] += 1.

    def train(self, nums_iter: List[int], update_frequency: int):
        self.update_frequency = update_frequency
        self._setup_history(nums_iter // self.update_frequency)
        self._init_params()
        for i in range(nums_iter[0]):
            self.train_iter = i
            update = False
            if i % self.update_frequency == 0:
                update = True
            self._step1(update=update)
        for i in range(nums_iter[1]):
            self.train_iter = i
            update = False
            if i % self.update_frequency == 0:
                update = True
            self._step2(update=update)

    def _step1(self, update, DQ=None, DR=None):
        temp_Q = self.Q.copy()
        temp_R = self.R.copy()

        dR = self.lr_w1 / self.N / sqrt2pi * (np.diag(np.sqrt(
            self.S)) + np.diag(self.R) / np.diag(np.sqrt(self.Q))) * np.power(
                self.P_task, self.seq_length - 1)
        dQ = self.lr_w1 / self.N * np.sqrt(2 / np.pi) * (np.sqrt(np.diag(
            self.Q)) + np.diag(self.R) / np.diag(np.sqrt(self.S))) * np.power(
                self.P_task,
                self.seq_length - 1) + np.power(self.lr_w1, 2) * np.power(
                    self.P_task, self.seq_length) / self.seq_length / self.N

        temp_Q += np.diag(dQ)
        temp_R += np.diag(dR)
        temp_R /= np.sqrt(np.abs(temp_Q))
        temp_Q /= temp_Q

        self.Q = temp_Q
        self.R = temp_R

        if update:
            update_index = self.train_iter // self.update_frequency
            self.history['phase1']['Q'][update_index] = self.Q
            self.history['phase1']['R'][update_index] = self.R
            self.history['phase1']['P_tilde'][update_index] = self.P_tilde
            self.history['phase1']['P'][update_index] = self.P_task
            self.history['phase1']['overlap_tilde'][update_index] = self.overlap
            self.history['phase1']['overlap'][update_index] = self.overlap_task

    def _step2(self, update, DV=None, DQ=None, DR=None):

        temp_VS = self.VS.copy()
        temp_Q = self.Q.copy()
        temp_R = self.R.copy()
        if DV is not None:
            dV = DV
        else:
            dV = self.lr_v / sqrt2pi / self.N * np.power(
                self.P_tilde, self.seq_length -
                1) * (self.VS * np.diag(self.Q) / self.norm_student +
                      self.VT * np.diag(self.R) / self.norm_teacher)
        if DR is not None:
            dR = DR
        else:
            dR = self.lr_w2 / sqrt2pi / self.N * np.power(
                self.P_tilde, self.seq_length - 1) * self.VS[:, None] * (
                    self.VS * np.diag(self.R) / self.norm_student +
                    self.VT * np.diag(self.S) / self.norm_teacher)[:, None].T

        v_term = self.VS * np.diag(
            self.Q) / self.norm_student + self.VT * np.diag(
                self.R) / self.norm_teacher

        dQ = 2 * self.lr_w2 / sqrt2pi / self.N * np.power(
            self.P_tilde, self.seq_length - 1
        ) * self.VS[:, None] @ v_term[:, None].T + np.power(
            self.lr_w2, 2) * np.power(
                self.P_tilde, self.seq_length) / self.seq_length / self.N * (
                    self.VS[:, None] @ self.VS[:, None].T)

        temp_VS += dV
        if self.V_norm != 0:
            temp_VS *= np.sqrt(self.num_task) / np.linalg.norm(temp_VS)
        else:
            temp_VS /= np.linalg.norm(temp_VS)
        temp_R += np.diag(np.diag(dR))
        temp_Q += np.diag(np.diag(dQ))
        temp_R /= np.sqrt(np.abs(temp_Q))
        temp_Q /= temp_Q

        self.VS = temp_VS
        self.Q = temp_Q
        self.R = temp_R

        if update:
            update_index = self.train_iter // self.update_frequency
            self.history['phase2']['VS'][update_index] = self.VS
            self.history['phase2']['Q'][update_index] = self.Q
            self.history['phase2']['R'][update_index] = self.R
            self.history['phase2']['P_tilde'][update_index] = self.P_tilde
            self.history['phase2']['P'][update_index] = self.P_task
            self.history['phase2']['VSVT'][update_index] = np.dot(
                self.VT, self.VS) / np.linalg.norm(self.VT) / np.linalg.norm(
                    self.VS)
            self.history['phase2']['overlap_tilde'][update_index] = self.overlap
            self.history['phase2']['overlap'][update_index] = self.overlap_task

    @property
    def norm_student(self):
        return np.sqrt(np.sum(self.VS * self.VS * np.diag(self.Q)))

    @property
    def norm_teacher(self):
        return np.sqrt(np.sum(self.VT * self.VT * np.diag(self.S)))

    @property
    def overlap(self):
        return np.sum(self.VS * self.VT *
                      np.diag(self.R)) / self.norm_teacher / self.norm_student

    @property
    def P_tilde(self):

        return 1 - np.arccos(self.overlap) / np.pi

    @property
    def overlap_task(self):

        return np.diag(self.R) / np.sqrt(np.diag(self.Q)) / np.sqrt(
            np.diag(self.S))

    @property
    def P_task(self):

        return 1 - np.arccos(self.overlap_task) / np.pi


class HRLODESolverIdentical(HRLODESolver):

    def _step2(self, update, DV=None, DQ=None, DR=None):
        temp_VS = self.VS.copy()
        temp_Q = self.Q.copy()
        temp_R = self.R.copy()

        dV = self.lr_v / sqrt2pi / self.N * np.power(
            self.P_tilde,
            self.seq_length - 1) * (self.VS @ self.Q.T / self.norm_student +
                                    self.VT @ self.R.T / self.norm_teacher)
        dR = self.lr_w2 / sqrt2pi / self.N * np.power(
            self.P_tilde, self.seq_length - 1) * self.VS[:, None] @ (
                self.VS @ self.R / self.norm_student +
                self.VT @ self.S.T / self.norm_teacher)[:, None].T

        v_term = self.VS @ self.Q / self.norm_student + self.VT @ self.R.T / self.norm_teacher

        dQ = 2 * self.lr_w2 / sqrt2pi / self.N * np.power(
            self.P_tilde, self.seq_length - 1
        ) * self.VS[:, None] @ v_term[:, None].T + np.power(
            self.lr_w2, 2) * np.power(
                self.P_tilde, self.seq_length) / self.seq_length / self.N * (
                    self.VS[:, None] @ self.VS[:, None].T)

        temp_VS += dV
        temp_R += dR
        temp_Q += dQ

        self.VS = temp_VS
        self.Q = temp_Q
        self.R = temp_R

        if update:
            update_index = self.train_iter // self.update_frequency
            self.history['phase2']['VS'][update_index] = self.VS
            self.history['phase2']['Q'][update_index] = self.Q
            self.history['phase2']['R'][update_index] = self.R
            self.history['phase2']['P_tilde'][update_index] = self.P_tilde
            self.history['phase2']['P'][update_index] = self.P_task
            self.history['phase2']['VSVT'][update_index] = np.dot(
                self.VT, self.VS) / np.linalg.norm(self.VT) / np.linalg.norm(
                    self.VS)

    @property
    def P_tilde(self):

        angle = np.arccos(
            np.sum([self.VS[:, None] @ self.VT[:, None].T * self.R]) /
            self.norm_teacher / self.norm_student)

        return 1 - angle / np.pi

    @property
    def norm_student(self):

        return np.sqrt(np.sum(self.VS[:, None] @ self.VS[:, None].T * self.Q))

    @property
    def norm_teacher(self):
        return np.sqrt(np.sum(self.VT[:, None] @ self.VT[:, None].T * self.S))