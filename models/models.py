import torch
from torch import nn
from torch import optim
from torch import Tensor

from models.ffmodel import FFModel as FFM


class BPModel(nn.Module):
    def __init__(self, dims: list, optimizer: optim, lr: float, device=None) -> None:
        super(BPModel, self).__init__()

        self.layers = nn.Sequential(*self.__get_layers(dims, device))
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.loss_fc = nn.CrossEntropyLoss()

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
    
    def __get_layers(self, dims: list, device=None) -> list:
        layers = list()
        for dim in range(len(dims) - 1):
            layers.append(nn.Linear(dims[dim], dims[dim+1], device=device))
            layers.append(nn.ReLU())
        return layers
    
    def inference(self, input: Tensor) -> Tensor:
        return self.forward(input)\
        
    def update(self, x: Tensor, y: Tensor) -> None:
        y_hat = self.forward(x)
        loss = self.loss_fc(y_hat, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

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
    
    def update(self, pos_x: Tensor, pos_y: Tensor, neg_x: Tensor, neg_y: Tensor) -> None:
        pos_input = self.__combine_xy(pos_x, pos_y)
        neg_input = self.__combine_xy(neg_x, neg_y)

        pos_o, neg_o = pos_input, neg_input
        for layer in self.layers:
            pos_o, neg_o = layer.update(pos_o, neg_o)


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