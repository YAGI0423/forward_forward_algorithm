import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

def toTensor(x, device='cpu'):
    return torch.tensor(x, dtype=torch.float32, device=device)

def freeze_model(model, is_freeze: bool=True):
    for weight in model.parameters():
        weight.requires_grad = not is_freeze

def soft_update(origin_net, target_net, T=0.001):
    for ori_weight, target_weight in zip(origin_net.parameters(), target_net.parameters()):
        target_weight.data.copy_(T * ori_weight.data + (1. - T) * target_weight.data)

if __name__ == '__main__':
    ep = 0
    while True:

        #Test++++++++++++++++++++++++++++++++++++
        if ep % 10 == 0:
            test_time_step, test_rewards = replayOneEpisod(env, actor, buffer=None, action_size=None, device=device)
            print(f'\nTEST Reward: {test_rewards:.3f}, times: {test_time_step}')

            save_his(his_r, his_act, his_cri)
        #End+++++++++++++++++++++++++++++++++++++

    