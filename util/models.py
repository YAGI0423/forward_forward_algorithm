import torch
import torch. nn as nn

from torch import Tensor

class BPModel(nn.Module):
    def __init__(self, dims: list) -> None:
        super(BPModel, self).__init__()
        self.layers = nn.Sequential(
            *tuple(nn.Linear(dims[dim], dims[dim+1]) for dim in range(len(dims)-1))
        )

    def forward(self, input: Tensor) -> Tensor:
        pass
