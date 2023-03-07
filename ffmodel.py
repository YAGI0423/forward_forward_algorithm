import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F


class FFLinear(nn.Linear):
    __constants__ = ('in_features', 'out_features')
    in_features: int
    out_features: int
    weight: Tensor
        
    def __init__(self, in_feature: int, out_features: int, optimizer: torch.optim, lr: float,
                 bias: bool=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FFLinear, self).__init__(in_feature, out_features, bias, **factory_kwargs)
        
        self.activation = nn.ReLU()
        self.optim = optimizer(self.parameters(), lr=lr)
        self.threshold = 2.0
        
    def forward(self, input) -> Tensor:
        out = self.__layerNorm(input)
        out = F.linear(out, self.weight, self.bias)
        return self.activation(out)
    
    def update(self, pos_x, neg_x) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        pos_out = self.forward(pos_x).pow(exponent=2).mean(dim=1) #shape: (Batch, )
        neg_out = self.forward(neg_x).pow(exponent=2).mean(dim=1) #shape: (Batch, )
        
        loss = torch.cat([-pos_out + self.threshold, neg_out - self.threshold])
        loss = torch.log(1. + torch.exp(loss)).mean()
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss, (self.forward(pos_x).detach(), self.forward(neg_x).detach())
    
    def __layerNorm(self, input: Tensor, eps: float=1e-4) -> Tensor:
        '''
        + 참고 repository의 정규화 코드
        input / (input.norm(p=2, dim=1, keepdim=True) + 1e-4)
        
        ## https://github.com/mohammadpz/pytorch_forward_forward/blob/main/main.py
        '''
        mean_ = input.mean(dim=1, keepdim=True)
        var_ = input.var(dim=1, keepdim=True, unbiased=False) #unbiased True=(N-1), False=N
        return (input - mean_) / torch.sqrt(var_ + eps)
    

class FFModel(nn.Module):
    def __init__(self, dims: list, optimizer: optim, lr: float,
                 bias: bool=True, device=None, dtype=None) -> None:
        super(FFModel, self).__init__()
        self.layers = tuple(
            FFLinear(dims[dim], dims[dim+1], optimizer, lr, bias, device, dtype) for dim in range(len(dims)-1)
        )

    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.size(0)
        goodness = torch.zeros(batch_size).to(input.device)

        out = input
        for layer in self.layers:
            out = layer(out)
            goodness += out.pow(exponent=2).mean(dim=1)
        return goodness