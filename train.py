import gym
import pybullet_envs

import models
import replayBuffer

from copy import deepcopy
import numpy as np

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

def replayOneEpisod(env, actor, buffer, action_size:int, device):
    state = env.reset()
    done = False
    
    time_step, rewardByEpi = 0, 0
    actor.eval()
    with torch.no_grad():
        while not done:
            action = actor(toTensor(state).to(device))
            action = action.detach()
            if device == 'cuda':
                action = action.cpu()
            action = action.numpy()

            if action_size is not None:
                action += np.random.randn(action_size) * LAMBDA
            action = action.clip(-1., 1.)
            state_n, reward, done, _ = env.step(action)

            if buffer is not None:
                buffer.add_buffer(state=state, action=action, reward=reward.reshape(-1), state_n=state_n)
            state = state_n.copy()

            time_step += 1
            rewardByEpi += reward
    return time_step, rewardByEpi

def update_critic(critic, target_critic, target_actor, crt_opt, buf):
    s_i, act, r, s_ii = buf

    critic.train()
    crt_opt.zero_grad()

    y_i = r + GAMMA * target_critic(s_ii, target_actor(s_ii))
    q_i = critic(s_i, act)

    loss = F.mse_loss(q_i, y_i)
    loss.backward()
    crt_opt.step()
    return loss.item()

def update_actor(actor, critic, act_opt, buf):
    s_i, *_ = buf
    
    actor.train()
    critic.train()

    act_opt.zero_grad()
    freeze_model(critic)

    act_y_hat = actor(s_i)
    loss = -critic(s_i, act_y_hat)
    loss = loss.mean()
    
    loss.backward()
    act_opt.step()
    freeze_model(critic, is_freeze=False)
    return loss.item()


ENV_ID = 'MinitaurBulletEnv-v0'
RENDER = True

LEARNING_RATE = 0.001
BATCH_SIZE = 32


LAMBDA = 0.4
GAMMA = 0.99 #discount factor
TAU = 0.001

ACT_SIZE = 8

device = 'cuda'

if __name__ == '__main__':
    spec = gym.envs.registry.spec(ENV_ID)
    spec._kwargs['render'] = RENDER
    env = gym.make(ENV_ID)

    buffer = replayBuffer.Buffer(buffer_size=10000, batch_size=BATCH_SIZE, device=device)

    actor = models.Actor(input_shape=28, output_shape=ACT_SIZE).to(device)
    critic = models.Critic(state_shape=28, act_size=ACT_SIZE).to(device)
    tg_actor, tg_critic = deepcopy(actor).to(device), deepcopy(critic).to(device)

    # act_opt = optim.SGD(actor.parameters(), lr=LEARNING_RATE, weight_decay=1e-6, momentum=0.9)
    # crt_opt = optim.SGD(critic.parameters(), lr=LEARNING_RATE, weight_decay=1e-6, momentum=0.9)

    act_opt = optim.Adam(actor.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    crt_opt = optim.Adam(critic.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    
    for ep in range(100):
        #collect buffer++++++++++++++++++++++++++
        time_step, rewards = replayOneEpisod(env, actor, buffer, action_size=ACT_SIZE, device=device)
        #End+++++++++++++++++++++++++++++++++++++


        #train model+++++++++++++++++++++++++++++
        buffer.update()
        buf = buffer.get_batch()
        
        cri_loss = update_critic(critic, tg_critic, tg_actor, crt_opt, buf)
        act_loss = update_actor(actor, critic, act_opt, buf)


        soft_update(actor, tg_actor, T=TAU)
        soft_update(critic, tg_critic, T=TAU)
        #End+++++++++++++++++++++++++++++++++++++

        print(f'Reward: {rewards:.3f}, times: {time_step}, Cri: {cri_loss:.3f}, Act: {act_loss:.3f}')

        #Test++++++++++++++++++++++++++++++++++++
        if ep % 10 == 0:
            test_time_step, test_rewards = replayOneEpisod(env, actor, buffer=None, action_size=None, device=device)
            print(f'Reward: {test_rewards:.3f}, times: {test_time_step}')
        #End+++++++++++++++++++++++++++++++++++++
        print('=' * 100)


    env.close()
    