import abc
from typing import List
import numpy as np
import network

sqrt2pi = np.sqrt(2 * np.pi)


class ODESolver():

    def __init__(self, **kwargs):

        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

    def _init_params(self):
        pass

    def _step(self):
        pass

    def _setup_history(self):
        pass

    def _update_history(self, **kwargs):
        for k, v in kwargs.items():
            self.history[k].append(v)

    def train(self, num_iter: int, update_frequency: int):
        self._init_params()
        self._setup_history()
        for i in range(num_iter):
            update = False
            if i % update_frequency == 0:
                update = True
            self._step(update=update)


class RLPerceptronODESolver(ODESolver):

    def __init__(self, lr_positive: float, lr_negative: float, seq_length: int,
                 delta: float):
        super().__init__(lr_positive=lr_positive,
                         lr_negative=lr_negative,
                         seq_length=seq_length,
                         delta=delta)

    def _init_params(self):
        self.lr_sum = self.lr_negative + self.lr_positive
        self.Q = 1. + abs(np.random.normal(0, 1))
        self.R = np.random.normal(0, 1)

    def _step(self):

        self.P = 1 - (1 / np.pi) * np.arccos(self.R / np.sqrt(self.Q))

        dR = (self.lr_sum) / np.sqrt(
            2 * np.pi) * (1 + self.R / np.sqrt(self.Q)) * np.power(
                self.P, self.seq_length -
                1) - self.lr_negative * self.R * np.sqrt(2 / (np.pi * self.Q))
        dQ = (self.lr_sum) * np.sqrt(
            2 * self.Q / np.pi) * (1 + self.R / np.sqrt(self.Q)) * np.power(
                self.P, self.seq_length -
                1) - 2 * self.lr_negative * np.sqrt(2 * self.Q / np.pi) + (
                    self.lr_positive**2 -
                    self.lr_negative**2) / self.seq_length * np.power(
                        self.P,
                        self.seq_length) + self.lr_negative**2 / self.seq_length

        self.R += dR * self.delta
        self.Q += dQ * self.delta
        self.exp_reward = np.power(self.P, self.seq_length)

        self._update_history(R=self.R,
                             Q=self.Q,
                             P=self.P,
                             exp_reward=self.exp_reward)

    def _setup_history(self):
        self.history = {'R': [], 'Q': [], 'P': [], 'exp_reward': []}


class MultistepRLPerceptronODESolver(ODESolver):

    def __init__(self, lr_positive: float, lr_negative: float, seq_length: int,
                 num_chain: int, train_per_chain: int, threshold: bool,
                 delta: float, update_frequency: int):
        super().__init__(lr_positive=lr_positive,
                         lr_negative=lr_negative,
                         seq_length=seq_length,
                         delta=delta)

        self.num_chain = num_chain
        self.train_per_chain = train_per_chain
        self.update_frequency = update_frequency
        self.threshold = threshold
        self.tot_length = self.num_chain * self.seq_length

    def init_params(self):
        self.teachers = []
        self.lr_sum = self.lr_negative + self.lr_positive
        for i in range(self.num_chain):
            """
            teacher_network = network.ContinuousTeacher(input_dimension=int(
                1 / self.delta),
                                                        hidden_dimensions=[1],
                                                        nonlinearity='sign')
            student_network = network.ContinuousStudent(input_dimension=int(
                1 / self.delta),
                                                        hidden_dimensions=[1],
                                                        nonlinearity='sign')
            QT = (teacher_network.layers[0].weight.data
                  @ teacher_network.layers[0].weight.data.T).item() * self.delta
            Q = (student_network.layers[0].weight.data
                 @ student_network.layers[0].weight.data.T).item() * self.delta
            teacher_network.layers[0].weight.data /= np.sqrt(QT)
            self.teachers.append(teacher_network.layers[0].weight.data.numpy())
            R = (teacher_network.layers[0].weight.data
                 @ student_network.layers[0].weight.data.T).item() * self.delta
            """
            Q = 1.
            QT = 1.
            R = 0.

            P = 1 - (1 / np.pi) * np.arccos(R / np.sqrt(Q))
            setattr(self, f"Q{i}_init", Q)
            setattr(self, f"R{i}_init", R)
            setattr(self, f"P{i}_init", P)
            setattr(self, f"QT{i}_int", QT)

        self.chain_index = 0

    def train(self, num_iter: int, flag: str):
        self.train_iter = 0
        self._setup_history()
        if flag == 'chained':
            for i in range(int(num_iter // self.train_per_chain)):
                self.chain_iter = 0
                self.chain_index = i
                for j in range(self.train_per_chain):
                    self._step_chained()
                    P = getattr(self, f"P{self.chain_index}")
                    if self.threshold and P >= 0.99:
                        break
        elif flag == 'baseline':
            for i in range(num_iter):
                self._step_baseline()

    def _step_chained(self):

        if self.chain_iter == 0:
            P = getattr(self, f"P{self.chain_index}_init")
            Q = getattr(self, f"Q{self.chain_index}_init")
            R = getattr(self, f"R{self.chain_index}_init")

        else:

            P = getattr(self, f"P{self.chain_index}")
            Q = getattr(self, f"Q{self.chain_index}")
            R = getattr(self, f"R{self.chain_index}")

        dR = (self.lr_sum) / np.sqrt(
            2 * np.pi) * (1 + R / np.sqrt(Q)) * np.power(
                P, self.seq_length - 1) - self.lr_negative * R * np.sqrt(
                    2 / (np.pi * Q))
        dQ = (self.lr_sum) * np.sqrt(
            2 * Q / np.pi) * (1 + R / np.sqrt(Q)) * np.power(
                P, self.seq_length -
                1) - 2 * self.lr_negative * np.sqrt(2 * Q / np.pi) + (
                    self.lr_positive**2 - self.lr_negative**2
                ) / self.seq_length * np.power(
                    P, self.seq_length) + self.lr_negative**2 / self.seq_length

        R += self.delta * dR
        Q += self.delta * dQ
        P = 1 - (1 / np.pi) * np.arccos(R / np.sqrt(Q))
        setattr(self, f"Q{self.chain_index}", Q)
        setattr(self, f"R{self.chain_index}", R)
        setattr(self, f"P{self.chain_index}", P)

        not_learned = np.power(
            0.5, self.seq_length * (self.num_chain - self.chain_index - 1))
        learned = 1
        for i in range(self.chain_index):
            learned *= np.power(getattr(self, f"P{i}"), self.seq_length)
        self.exp_reward = not_learned * learned * np.power(P, self.seq_length)
        if self.train_iter % self.update_frequency == 0:
            self._update_history(exp_reward=self.exp_reward)
        self.train_iter += 1
        self.chain_iter += 1

    def _step_baseline(self):
        exp_reward = 1
        for i in range(self.num_chain):
            if self.train_iter == 0:
                P = getattr(self, f"P{i}_init")
                Q = getattr(self, f"Q{i}_init")
                R = getattr(self, f"R{i}_init")
            else:
                P = getattr(self, f"P{i}")
                Q = getattr(self, f"Q{i}")
                R = getattr(self, f"R{i}")

            dR = (self.lr_sum) / np.sqrt(
                2 * np.pi) * (1 + R / np.sqrt(Q)) * np.power(
                    P, self.tot_length - 1) - self.lr_negative * R * np.sqrt(
                        2 / (np.pi * Q))

            dQ = (self.lr_sum) * np.sqrt(
                2 * Q / np.pi) * (1 + R / np.sqrt(Q)) * np.power(
                    P, self.tot_length -
                    1) - 2 * self.lr_negative * np.sqrt(2 * Q / np.pi) + (
                        self.lr_positive**2 - self.lr_negative**2
                    ) / self.seq_length * np.power(
                        P,
                        self.tot_length) + self.lr_negative**2 / self.seq_length

            R += self.delta * dR
            Q += self.delta * dQ
            P = 1 - (1 / np.pi) * np.arccos(R / np.sqrt(Q))
            setattr(self, f"Q{i}", Q)
            setattr(self, f"R{i}", R)
            setattr(self, f"P{i}", P)
            exp_reward *= np.power(P, self.seq_length)
        self.exp_reward = exp_reward
        if self.train_iter % self.update_frequency == 0:
            self._update_history(exp_reward=exp_reward,)
        self.train_iter += 1

    def _setup_history(self):
        self.history = {'exp_reward': []}


class BaselineRLPerceptronODESolver(ODESolver):

    def __init__(self, lr_positive: float, lr_negative: float, seq_length: int,
                 num_chain: int, total_train_step: int, delta: float,
                 update_frequency: int):
        super().__init__(lr_positive=lr_positive,
                         lr_negative=lr_negative,
                         seq_length=seq_length,
                         delta=delta)

        self.num_chain = num_chain
        self.total_train_step = total_train_step
        self.update_frequency = update_frequency
        self.tot_length = self.seq_length * self.num_chain

    def _init_params(self):
        self.lr_sum = self.lr_negative + self.lr_positive
        self.teachers = []
        for i in range(self.num_chain):
            teacher_network = network.ContinuousTeacher(input_dimension=self.D,
                                                        hidden_dimensions=[1],
                                                        nonlinearity='sign')
            student_network = network.ContinuousStudent(input_dimension=self.D,
                                                        hidden_dimensions=[1],
                                                        nonlinearity='sign')

            QT = (teacher_network.layers[0].weight.data
                  @ teacher_network.layers[0].weight.data.T).item() * self.delta
            Q = (student_network.layers[0].weight.data
                 @ student_network.layers[0].weight.data.T).item() * self.delta
            teacher_network.layers[0].weight.data /= np.sqrt(QT)
            self.teachers.append(teacher_network.layers[0].weight.data.numpy())
            R = (teacher_network.layers[0].weight.data
                 @ student_network.layers[0].weight.data.T).item() * self.delta

            P = 1 - (1 / np.pi) * np.arccos(R / np.sqrt(Q))
            setattr(self, f"Q{i}", Q)
            setattr(self, f"R{i}", R)
            setattr(self, f"P{i}", P)
            setattr(self, f"QT{i}", QT)
        self.train_iter = 0

    def _step(self):

        exp_reward = 1
        for i in range(self.num_chain):
            P = getattr(self, f"P{i}")
            Q = getattr(self, f"Q{i}")
            R = getattr(self, f"R{i}")

            dR = (self.lr_sum) / np.sqrt(
                2 * np.pi) * (1 + R / np.sqrt(Q)) * np.power(
                    P, self.tot_length - 1) - self.lr_negative * R * np.sqrt(
                        2 / (np.pi * Q))

            dQ = (self.lr_sum) * np.sqrt(
                2 * Q / np.pi) * (1 + R / np.sqrt(Q)) * np.power(
                    P, self.tot_length -
                    1) - 2 * self.lr_negative * np.sqrt(2 * Q / np.pi) + (
                        self.lr_positive**2 - self.lr_negative**2
                    ) / self.seq_length * np.power(
                        P,
                        self.tot_length) + self.lr_negative**2 / self.seq_length

            R += self.delta * dR
            Q += self.delta * dQ
            P = 1 - (1 / np.pi) * np.arccos(R / np.sqrt(Q))
            setattr(self, f"Q{i}", Q)
            setattr(self, f"R{i}", R)
            setattr(self, f"P{i}", P)
            exp_reward *= np.power(P, self.seq_length)
        self.exp_reward = exp_reward
        if self.train_iter % self.update_frequency == 0:
            self._update_history(exp_reward=exp_reward,)
        self.train_iter += 1

    def _setup_history(self):
        self.history = {
            'exp_reward': [],
        }


class HierarchicalODESolver(ODESolver):

    def __init__(self, VS: np.array, VT: np.array, students, teachers,
                 lr_ws: List[float], lr_v: float, seq_length: int, N: int):
        super().__init__()
        self.num_task = len(VS)
        self.seq_length = seq_length
        self.VS = VS
        self.VT = VT
        self.N = N
        self.lr_w1 = lr_ws[0]
        self.lr_w2 = lr_ws[1]
        self.lr_v = lr_v
        self.students = students
        self.teachers = teachers
        self._setup_history()
        self._init_params()

    def _setup_history(self):
        self.history = {
            'phase1': {
                'Q': [],
                'R': [],
                'P': [],
                'P_tilde': [],
                'VS': [],
                'VT':[]
            },
            'phase2': {
                'VT':[],
                'VS': [],
                'Q': [],
                'R': [],
                'P': [],
                'P_tilde': [],
            }
        }

    def _init_params(self):
        self.Q = np.zeros(shape=(self.num_task, self.num_task))
        self.R = np.zeros(shape=(self.num_task, self.num_task))
        self.S = np.zeros(shape=(self.num_task, self.num_task))

        if self.students is not None and self.teachers is not None:
            for i in range(self.num_task):
                for j in range(self.num_task):
                    self.Q[i][j] = (self.students[i].layers[0].weight.data
                                    @ self.students[j].layers[0].weight.data.T
                                   ).item() / self.N
                    self.R[i][j] = (self.students[i].layers[0].weight.data
                                    @ self.teachers[j].layers[0].weight.data.T
                                   ).item() / self.N
                    self.S[i][j] = (self.teachers[i].layers[0].weight.data
                                    @ self.teachers[j].layers[0].weight.data.T
                                   ).item() / self.N
        else:
            for i in range(self.num_task):
                self.Q[i][i] += 1.
                self.S[i][i] += 1.

        self.history['phase1']['Q'].append(self.Q)
        self.history['phase1']['R'].append(self.R)
        self.history['phase1']['P'].append(
            [self.P_task(k) for k in range(self.num_task)])
        self.history['phase1']['VS'].append(self.VS)
        self.history['phase1']['VT'].append(self.VT)

    def train(self, nums_iter: List[int], update_frequency: int):
        for i in range(nums_iter[0]):
            update = False
            if i % update_frequency == 0:
                update = True
            self._step1(update=update)
        for i in range(nums_iter[1]):
            update = False
            if i % update_frequency == 0:
                update = True
            self._step2(update=update)

    def _step1(self, update, DQ=None, DR=None):
        temp_Q = self.Q.copy()
        temp_R = self.R.copy()

        for k in range(self.num_task):
            if DR is None:
                dR = self.lr_w1 / self.N / sqrt2pi * (
                    1 + self.R[k][k] / np.sqrt(self.Q[k][k])) * np.power(
                        self.P_task(k), self.seq_length - 1)

            else:
                dR = DR[k][k]
            if DQ is None:
                dQ = self.lr_w1 / self.N * np.sqrt(2 * self.Q[k][k] / np.pi) * (
                    1 + self.R[k][k] / np.sqrt(self.Q[k][k])) * np.power(
                        self.P_task(k), self.seq_length - 1) + np.power(
                            self.lr_w1, 2) * np.power(self.P_task(
                                k), self.seq_length) / self.seq_length / self.N
            else:
                dQ = DQ[k][k]

            temp_R[k][k] += dR
            temp_Q[k][k] += dQ
        self.Q = temp_Q
        self.R = temp_R
        if update:
            self.history['phase1']['Q'].append(self.Q)
            self.history['phase1']['R'].append(self.R)
            self.history['phase1']['P_tilde'].append(self.P_tilde)
            self.history['phase1']['P'].append(
                [self.P_task(k) for k in range(self.num_task)])
            self.history['phase1']['VS'].append(self.VS)

    def _step2(self, update, DV=None, DQ=None, DR=None):
        temp_VS = self.VS.copy()
        temp_Q = self.Q.copy()
        temp_R = self.R.copy()
        for k in range(self.num_task):
            if DV is None:
                dV = self.lr_v / sqrt2pi / self.N * (
                    self.VS[k] * self.Q[k][k] / self.norm_student +
                    self.VT[k] * self.R[k][k] / self.norm_teacher) * np.power(
                        self.P_tilde, self.seq_length - 1)

                temp_VS[k] += dV
            else:
                temp_VS[k] += DV[k]

            if DR is None:
                dR = self.lr_w2 / sqrt2pi / self.N * self.VS[k] * np.power(
                    self.P_tilde, self.seq_length - 1) * (
                        (self.S[k][k] * self.VT[k] / self.norm_teacher) +
                        (self.R[k][k] * self.VS[k] / self.norm_student))
                temp_R[k][k] += dR
            else:
                temp_R[k][k] += DR[k][k]
            if DQ is None:

                vk_term = self.VS[k] * self.Q[k][k] / self.norm_student + self.R[
                    k][k] * self.VT[k] / self.norm_teacher

                vkvl_term = 1.

                dQ_k = self.lr_w2 / self.N / sqrt2pi * (
                    self.VS[k] * vk_term) * np.power(self.P_tilde,
                                                     self.seq_length - 1)

                dQ_kl = np.power(
                    self.lr_w2, 2) / self.seq_length / self.N * np.power(
                        self.P_tilde,
                        self.seq_length) * self.VS[k] * self.VS[k] * vkvl_term

                temp_Q[k][k] += 2 * dQ_k + dQ_kl

            else:
                temp_Q[k][k] += DQ[k][k]

        self.VS = temp_VS
        self.Q = temp_Q
        self.R = temp_R

        if update:
            self.history['phase2']['VS'].append(self.VS)
            self.history['phase2']['Q'].append(self.Q)
            self.history['phase2']['R'].append(self.R)
            self.history['phase2']['P_tilde'].append(self.P_tilde)
            self.history['phase2']['P'].append(
                [self.P_task(k) for k in range(self.num_task)])

    @property
    def norm_student(self):
        return np.sqrt(
            np.sum(self.VS[i] * self.VS[i] * self.Q[i][i]
                   for i in range(self.num_task)))

    @property
    def norm_teacher(self):
        return np.sqrt(
            np.sum(self.VT[i] * self.VT[i] * self.S[i][i]
                   for i in range(self.num_task)))

    @property
    def P_tilde(self):

        angle = np.arccos(
            sum([
                self.VS[i] * self.VT[i] * self.R[i][i]
                for i in range(self.num_task)
            ]) / self.norm_teacher / self.norm_student)

        return 1 - angle / np.pi

    def P_task(self, task_index):

        angle = np.arccos(self.R[task_index, task_index] /
                          np.sqrt(self.Q[task_index, task_index]) /
                          np.sqrt(self.S[task_index, task_index]))
        return 1 - angle / np.pi


class HierarchicalODESolverIdentical(HierarchicalODESolver):

    def _step2(self, update, DV=None, DQ=None, DR=None):
        temp_VS = self.VS.copy()
        temp_Q = self.Q.copy()
        temp_R = self.R.copy()

        for k in range(self.num_task):
            if DV is None:
                dV = self.lr_v / sqrt2pi / np.power(self.N, 1) * (np.sum([
                    self.VS[i] * self.Q[k][i] for i in range(self.num_task)
                ]) / self.norm_student + np.sum([
                    self.VT[i] * self.R[k][i] for i in range(self.num_task)
                ]) / self.norm_teacher) * np.power(self.P_tilde,
                                                   self.seq_length - 1)

                temp_VS[k] += dV
            else:
                temp_VS[k] += DV[k]

            for l in range(self.num_task):
                if DR is None:
                    dR = self.lr_w2 / sqrt2pi / self.N * self.VS[k] * np.power(
                        self.P_tilde, self.seq_length - 1) * (sum([
                            self.S[l][i] * self.VT[i]
                            for i in range(self.num_task)
                        ] / self.norm_teacher) + sum([
                            self.R[i][l] * self.VS[i]
                            for i in range(self.num_task)
                        ] / self.norm_student))
                    temp_R[k][l] += dR
                else:
                    temp_R[k][l] += DR[k][l]

                if DQ is None:

                    vk_term = sum([
                        self.VS[i] * self.Q[i][l] for i in range(self.num_task)
                    ]) / self.norm_student + sum([
                        self.R[l][i] * self.VT[i] for i in range(self.num_task)
                    ]) / self.norm_teacher
                    vl_term = sum([
                        self.VS[i] * self.Q[i][k] for i in range(self.num_task)
                    ]) / self.norm_student + sum([
                        self.R[k][i] * self.VT[i] for i in range(self.num_task)
                    ]) / self.norm_teacher

                    vkvl_term = 1.

                    dQ_k = self.lr_w2 / self.N / sqrt2pi * (
                        self.VS[k] * vk_term) * np.power(
                            self.P_tilde, self.seq_length - 1)

                    dQ_l = self.lr_w2 / self.N / sqrt2pi * (
                        self.VS[l] * vl_term) * np.power(
                            self.P_tilde, self.seq_length - 1)

                    dQ_kl = np.power(self.lr_w2,
                                     2) / self.seq_length / self.N * np.power(
                                         self.P_tilde, self.seq_length
                                     ) * self.VS[k] * self.VS[l] * vkvl_term

                    temp_Q[k][l] += dQ_k + dQ_l + dQ_kl

                else:
                    temp_Q[k][l] += DQ[k][l]

        self.VS = temp_VS
        self.Q = temp_Q
        self.R = temp_R

        if update:
            self.history['phase2']['VS'].append(self.VS)
            self.history['phase2']['Q'].append(self.Q)
            self.history['phase2']['R'].append(self.R)
            self.history['phase2']['P_tilde'].append(self.P_tilde)
            self.history['phase2']['P'].append(
                [self.P_task(k) for k in range(self.num_task)])

    @property
    def P_tilde(self):

        angle = np.arccos(
            sum([
                self.VS[i] * self.VT[j] * self.R[i][j]
                for i in range(self.num_task)
                for j in range(self.num_task)
            ]) / self.norm_teacher / self.norm_student)

        return 1 - angle / np.pi

    @property
    def norm_student(self):

        return np.sqrt(
            np.sum(self.VS[i] * self.VS[j] * self.Q[i][j]
                   for i in range(self.num_task)
                   for j in range(self.num_task)))

    @property
    def norm_teacher(self):
        return np.sqrt(
            np.sum(self.VT[i] * self.VT[j] * self.S[i][j]
                   for i in range(self.num_task)
                   for j in range(self.num_task)))
