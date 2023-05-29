import abc
import os
from datetime import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional
from typing import Union

import torch
import torch.nn as nn


@dataclass
class BaseSolver(abc.ABC):
    criterion: Union[nn.Module, None]
    optimizer_type: Union[str, None]
    lr: Union[float, None]
    logdir: str

    def __post_init__(self):
        pass

    @abc.abstractmethod
    def _setup_train(self, **kwargs):
        return NotImplementedError

    @abc.abstractmethod
    def _step(self, **kwargs):
        return NotImplementedError()

    @abc.abstractmethod
    def inference(self, **kwargs):
        return NotImplementedError

    @abc.abstractmethod
    def train(self, **kwargs):
        return NotImplementedError()

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
