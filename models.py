import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(input_shape, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, output_shape)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        return out

class Critic(nn.Module):
    def __init__(self, state_shape, act_size):
        super(Critic, self).__init__()
        self.stateLayer = nn.Linear(state_shape, 400)
        self.outLayer1 = nn.Linear(400 + act_size, 300)
        self.outLayer2 = nn.Linear(300, 1)

    def forward(self, state, action):
        s_x = F.relu(self.stateLayer(state))
        cat = torch.cat([s_x, action], dim=1)
        out = F.relu(self.outLayer1(cat))
        out = self.outLayer2(out)
        return out
