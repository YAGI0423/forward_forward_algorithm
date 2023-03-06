import torch
import torch. nn as nn
from torch import optim
from torch import Tensor

from ffmodel import FFModel as FFM


class BPModel(nn.Module):
    def __init__(self, dims: list, optimizer: optim, lr: float, device=None) -> None:
        super(BPModel, self).__init__()

        self.layers = nn.Sequential(*self.__get_layers(dims, device))
        self.optimizer = optimizer(self.parameters(), lr=lr)

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
    
    def __get_layers(self, dims: list, device=None) -> list:
        layers = list()
        for dim in range(len(dims) - 1):
            layers.append(nn.Linear(dims[dim], dims[dim+1], device=device))
            layers.append(nn.ReLU())
        return layers
    
    def inference(self, input: Tensor) -> Tensor:
        return self.forward(input)
    

class FFModel(FFM):
    def __init__(self, dims: list, optimizer: optim, lr: float, device=None) -> None:
        super(FFModel, self).__init__(dims, optimizer, lr, device=device)
        self.CLASS_NUM = 10

    def inference(self, input: Tensor) -> Tensor:
        #input shape: Batch x height*width(784)
        batch_size = input.size(0)
        
        input = input.unsqueeze(1).repeat(1, self.CLASS_NUM, 1) #Batch x CLASS_NUM x height*width
        input = input.view(batch_size * self.CLASS_NUM, -1) #Batch*CLASS_NUM x 784

        y_pixel = torch.arange(self.CLASS_NUM).repeat(batch_size) #y_pixel: (Batch*CLASS_NUM, )
        
        input_ = self.__combine_xy(input, y_pixel)

        goodness = self.forward(input_).view(batch_size, -1)
        return goodness


    def __combine_xy(self, x: Tensor, y: Tensor) -> Tensor:
        '''
        forward-forward Model 입력 데이터 반환

        X shape: Batch x 784(Ch * Height * Width)
        Y shape: Batch x Label
        '''
        batch_size = y.size(0)

        x_ = x.clone()
        x_[:, :self.CLASS_NUM] = 0.
        x_[range(batch_size), y] = x_.max()
        return x_