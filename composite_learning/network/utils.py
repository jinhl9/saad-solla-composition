import torch


def heaviside(x):
    return torch.heaviside(x, values=torch.tensor([0.]))
