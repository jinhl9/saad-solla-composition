import torch
import torch.nn as nn


def heaviside(x):
    return torch.heaviside(x, values=torch.tensor([0.]))


def scaled_sigmoid(x):
    sigmoid = nn.Sigmoid()
    return (sigmoid(x) - 0.5) * 2
