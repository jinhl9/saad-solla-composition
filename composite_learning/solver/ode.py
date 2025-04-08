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