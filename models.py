import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, epsilon=0.3):
        super(Actor, self).__init__()
        self.epsilon = epsilon

        self.layer1 = nn.Linear(input_shape, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, output_shape)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = torch.tanh(self.layer3(out))

        # out += self.epsilon * torch.randn(size=out.shape)
        # out = torch.clip(out, -1, 1)
        return out

class Critic(nn.Module):
    def __init__(self, state_shape, act_size, n_atoms, v_min, v_max):
        super(Critic, self).__init__()
        self.stateLayer = nn.Linear(state_shape, 400)
        self.outLayer1 = nn.Linear(400 + act_size, 300)
        self.outLayer2 = nn.Linear(300, n_atoms)

        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer('supports', torch.arange(v_min, v_max+delta, delta)) #'supports'가 key인 버퍼

    def forward(self, state, action):
        s_x = F.relu(self.stateLayer(state))
        cat = torch.cat([s_x, action], dim=1)
        out = F.relu(self.outLayer1(cat))
        out = self.outLayer2(out)
        return out
    
    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)
