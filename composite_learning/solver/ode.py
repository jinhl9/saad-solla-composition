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

    def train(self, num_iter: int):
        self._init_params()
        self._setup_history()
        for _ in range(num_iter):
            self._step()


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
