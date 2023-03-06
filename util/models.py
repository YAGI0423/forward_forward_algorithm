import torch
import torch. nn as nn
from torch.nn import functional as F

from torch import Tensor

class BPModel(nn.Module):
    def __init__(self, dims: list, optimizer, lr) -> None:
        super(BPModel, self).__init__()

        self.layers = nn.Sequential(*self.__get_layers(dims))
        self.optim = optimizer(self.parameters(), lr=lr)

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
    
    def __get_layers(self, dims: list) -> list:
        layers = list()
        for dim in range(len(dims) - 1):
            layers.append(nn.Linear(dims[dim], dims[dim+1]))
            layers.append(nn.ReLU())
        return layers