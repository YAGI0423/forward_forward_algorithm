import torch. nn as nn
from torch import optim

from torch import Tensor

class BPModel(nn.Module):
    def __init__(self, dims: list, optimizer: optim, lr: float) -> None:
        super(BPModel, self).__init__()

        self.layers = nn.Sequential(*self.__get_layers(dims))
        self.optimizer = optimizer(self.parameters(), lr=lr)

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
    
    def __get_layers(self, dims: list) -> list:
        layers = list()
        for dim in range(len(dims) - 1):
            layers.append(nn.Linear(dims[dim], dims[dim+1]))
            layers.append(nn.ReLU())
        return layers