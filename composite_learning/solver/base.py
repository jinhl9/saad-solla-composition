import abc
import os
import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BaseSolver(abc.ABC):
    criterion: nn.Module
    optimizer_type: str
    lr: float
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
        filename = datetime.now().strftime("%Y%m%d%H%M%S")
        self.logger = logging.getLogger()
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
        handler.setLevel(logging.INFO)
        file_handler = logging.FileHandler(
            os.path.join(self.logdir, f"{filename}.log"))
        file_handler.setFormattter(
            logging.Formatter("%(asctimes|%(levelname)s|%(name)s|%(message)s"))
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)
