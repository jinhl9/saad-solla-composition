import abc
import numpy as np
import network


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
                 lr_w: float, lr_v: float, seq_length: int, delta: float,
                 N: int):
        super().__init__(lr_positive=lr_w, seq_length=seq_length, delta=delta)
        self.num_task = len(VS)
        self.VS = VS
        self.VT = VT
        self.N = N
        self.lr_w = lr_w
        self.lr_v = lr_v
        self.dVs = []
        self.dQs = []
        self.dRs = []
        self.students = students
        self.teachers = teachers
        self._setup_history()
        self._init_params()

    def _setup_history(self):
        self.history = {
            'VS': [],
            'Q': [],
            'Qk': [],
            'Ql': [],
            'Qkl': [],
            'R': [],
            'P': [],
            'norm_student': [],
            'norm_teacher': []
        }

    def _product_norm(self, x: np.array, y: np.array):
        return np.linalg.norm(
            np.array([sum(x[i] * y[i]) for i in range(self.num_task)]))

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

    @property
    def overlap(self):

        angle = np.arccos(
            sum([
                self.VS[i] * self.VT[j] * self.R[i][j]
                for i in range(self.num_task)
                for j in range(self.num_task)
            ]) / self.norm_teacher / self.norm_student)

        return 1 - angle / np.pi

    def _init_params(self):
        self.Q = np.zeros(shape=(self.num_task, self.num_task))
        self.Qk = np.zeros(shape=(self.num_task, self.num_task))
        self.Ql = np.zeros(shape=(self.num_task, self.num_task))
        self.Qkl = np.zeros(shape=(self.num_task, self.num_task))
        self.R = np.zeros(shape=(self.num_task, self.num_task))
        self.S = np.zeros(shape=(self.num_task, self.num_task))

        if self.students is not None and self.teachers is not None:
            for i in range(self.num_task):
                for j in range(self.num_task):
                    self.Q[i][j] = (self.students[i].layers[0].weight.data
                                    @ self.students[j].layers[0].weight.data.T
                                   ).item() / self.N
                    self.Qk[i][j] = (self.students[i].layers[0].weight.data
                                     @ self.students[j].layers[0].weight.data.T
                                    ).item() / self.N
                    self.Ql[i][j] = (self.students[i].layers[0].weight.data
                                     @ self.students[j].layers[0].weight.data.T
                                    ).item() / self.N
                    self.Qkl[i][j] = (self.students[i].layers[0].weight.data
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
        self.P = self.overlap
        self.history['VS'].append(self.VS)
        self.history['Q'].append(self.Q)
        self.history['Qk'].append(self.Qk)
        self.history['Ql'].append(self.Ql)
        self.history['Qkl'].append(self.Qkl)
        self.history['R'].append(self.R)
        self.history['P'].append(self.P)

    def _step(self, update, DV, DQ, DR):
        temp_VS = self.VS.copy()
        temp_Q = self.Q.copy()
        temp_R = self.R.copy()
        _dvs = []
        for k in range(self.num_task):
            if DV is None:
                dV = self.lr_v / np.sqrt(2 * np.pi * self.N) * (np.sum([
                    self.VT[i] * self.R[k][i] for i in range(self.num_task)
                ]) / self.norm_teacher + np.sum([
                    self.VS[i] * self.Q[i][k] for i in range(self.num_task)
                ]) / self.norm_student) * np.power(self.P, self.seq_length - 1)
                temp_VS[k] += dV
                _dvs.append(dV)
            else:
                temp_VS[k] += DV[k]
                _dvs.append(DV[k])
            for l in range(self.num_task):
                if DR is None:
                    dR = self.lr_w / np.sqrt(
                        2 * np.pi) / self.N * self.VS[k] * np.power(
                            self.P, self.seq_length - 1) * ((np.sum([
                                self.VT[i] * self.S[i][l]
                                for i in range(self.num_task)
                            ]) / self.norm_teacher) + ((np.sum([
                                self.VS[i] * self.R[i][l]
                                for i in range(self.num_task)
                            ])) / self.norm_student))
                    temp_R[k][l] += dR
                    if k == 0 and l == 0:
                        r = dR
                else:
                    temp_R[k][l] += DR[k][l]
                    r = DR[0][0]
                if DQ is None:

                    vk_term = np.sum([
                        self.VT[i] * self.R[l][i] for i in range(self.num_task)
                    ]) / self.norm_teacher + np.sum([
                        self.VS[i] * self.Q[i][l] for i in range(self.num_task)
                    ]) / self.norm_student

                    vl_term = np.sum([
                        self.VT[i] * self.R[k][i] for i in range(self.num_task)
                    ]) / self.norm_teacher + np.sum([
                        self.VS[i] * self.Q[i][k] for i in range(self.num_task)
                    ]) / self.norm_student

                    if k == l:
                        vkvl_term = 1.
                    else:
                        vkvl_term = 1.

                    dQ = self.lr_w / self.N / np.sqrt(2 * np.pi) * (
                        self.VS[k] * vk_term + self.VS[l] * vl_term) * np.power(
                            self.P, self.seq_length - 1)
                    +np.power(self.lr_w,
                              2) / self.seq_length / self.N * np.power(
                                  self.P, 2 * self.seq_length
                              ) * self.VS[k] * self.VS[l] * vkvl_term

                    temp_Q[k][l] += dQ

                    if k == 0 and l == 0:
                        q = dQ
                else:
                    temp_Q[k][l] += DQ[k][l]
                    q = DQ[0][0]

        self.VS = temp_VS
        self.Q = temp_Q
        self.R = temp_R
        self.P = self.overlap
        self.dVs.append(_dvs)
        self.dQs.append(q)
        self.dRs.append(r)
        if update:
            self.history['VS'].append(self.VS)
            self.history['Q'].append(self.Q)
            self.history['R'].append(self.R)
            self.history['P'].append(self.P)
            self.history['norm_teacher'].append(self.norm_teacher)
            self.history['norm_student'].append(self.norm_student)


class HierarchicalODESolver2(HierarchicalODESolver):

    def _step(self, update, DV, DQ, DR):
        temp_VS = self.VS.copy()
        temp_Q = self.Q.copy()
        temp_R = self.R.copy()
        temp_Qk = self.Qk.copy()
        temp_Ql = self.Ql.copy()
        temp_Qkl = self.Qkl.copy()
        _dvs = []
        for k in range(self.num_task):
            if DV is None:

                dV = self.lr_v / np.sqrt(2 * np.pi) / np.power(self.N, 1) * (
                    self.VS[k] * self.Q[k][k] / self.norm_student +
                    self.VT[k] * self.R[k][k] / self.norm_teacher) * np.power(
                        self.P, self.seq_length - 1)
                """
                dV = self.lr_v / np.sqrt(2 * np.pi) / np.power(
                    self.N, 1) * (np.sum([
                        self.VT[i] * self.R[k][i] for i in range(self.num_task)
                    ]) / self.norm_teacher_tot + np.sum([
                        self.VS[i] * self.Q[i][k] for i in range(self.num_task)
                    ]) / self.norm_student_tot) * np.power(
                        self.P, self.seq_length - 1)
                """
                temp_VS[k] += dV
                _dvs.append(dV)
            else:
                temp_VS[k] += DV[k]
                _dvs.append(DV[k])
            for l in range(self.num_task):
                if DR is None:
                    dR = self.lr_w / np.sqrt(
                        2 * np.pi) / self.N * self.VS[k] * np.power(
                            self.P, self.seq_length - 1) * (
                                (self.S[l][l] * self.VT[l] / self.norm_teacher)
                                +
                                (self.R[l][l] * self.VS[l] / self.norm_student))
                    temp_R[k][l] += dR
                    if k == 0 and l == 0:
                        r = dR
                else:
                    temp_R[k][l] += DR[k][l]
                    r = DR[0][0]
                if DQ is None:

                    vk_term = self.VS[l] * self.Q[l][
                        l] / self.norm_student + self.R[l][l] * self.VT[
                            l] / self.norm_teacher
                    vl_term = self.VS[k] * self.Q[k][
                        k] / self.norm_student + self.R[k][k] * self.VT[
                            k] / self.norm_teacher
                    if k == l:
                        vkvl_term = 1.
                    else:
                        vkvl_term = 0.
                    """
                    dQ = self.lr_w / self.N / np.sqrt(2 * np.pi) * (
                        self.VS[k] * vk_term + self.VS[l] * vl_term) * np.power(
                            self.P, self.seq_length - 1)
                    +np.power(self.lr_w,
                              2) / self.seq_length / self.N * np.power(
                                  self.P, self.seq_length
                              ) * self.VS[k] * self.VS[l] * vkvl_term
                    """
                    dQ_k = self.lr_w / self.N / np.sqrt(
                        2 * np.pi) * (self.VS[k] * vk_term) * np.power(
                            self.P, self.seq_length - 1)

                    dQ_l = self.lr_w / self.N / np.sqrt(
                        2 * np.pi) * (self.VS[l] * vl_term) * np.power(
                            self.P, self.seq_length - 1)

                    dQ_kl = np.power(self.lr_w,
                                     2) / self.seq_length / self.N * np.power(
                                         self.P, self.seq_length
                                     ) * self.VS[k] * self.VS[l] * vkvl_term

                    temp_Q[k][l] += dQ_k + dQ_l + dQ_kl

                    temp_Qk[k][l] += dQ_k
                    temp_Ql[k][l] += dQ_l
                    temp_Qkl[k][l] += dQ_kl

                    if k == 0 and l == 0:
                        q = dQ_k + dQ_l + dQ_kl
                else:
                    temp_Q[k][l] += DQ[k][l]
                    q = DQ[0][0]

        self.VS = temp_VS
        self.Q = temp_Q
        self.Qk = temp_Qk
        self.Ql = temp_Ql
        self.Qkl = temp_Qkl
        self.R = temp_R
        self.P = self.overlap
        self.dVs.append(_dvs)
        self.dQs.append(q)
        self.dRs.append(r)
        if update:
            self.history['VS'].append(self.VS)
            self.history['Q'].append(self.Q)
            self.history['Qk'].append(self.Qk)
            self.history['Ql'].append(self.Ql)
            self.history['Qkl'].append(self.Qkl)
            self.history['R'].append(self.R)
            self.history['P'].append(self.P)
            self.history['norm_teacher'].append(self.norm_teacher)
            self.history['norm_student'].append(self.norm_student)

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
    def overlap(self):

        angle = np.arccos(
            sum([
                self.VS[i] * self.VT[i] * self.R[i][i]
                for i in range(self.num_task)
            ]) / self.norm_teacher / self.norm_student)

        return 1 - angle / np.pi

    @property
    def norm_student_tot(self):

        return np.sqrt(
            np.sum(self.VS[i] * self.VS[j] * self.Q[i][j]
                   for i in range(self.num_task)
                   for j in range(self.num_task)))

    @property
    def norm_teacher_tot(self):
        return np.sqrt(
            np.sum(self.VT[i] * self.VT[j] * self.S[i][j]
                   for i in range(self.num_task)
                   for j in range(self.num_task)))