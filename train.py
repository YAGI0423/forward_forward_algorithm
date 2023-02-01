import gym
import pybullet_envs

import models
import replayBuffer

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
                buffer.add_buffer(state=state, action=action, reward=reward.reshape(-1), state_n=state_n, done=done)
            state = state_n.copy()

            time_step += 1
            rewardByEpi += reward
    return time_step, rewardByEpi

# def distr_projection(q_ii, reward, is_done, gamma, device:str='cpu'):
#     batch_size = reward.size(0)
    
#     atom_tens = torch.arange(N_ATOMS, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1)

#     tz_j = torch.maximum(torch.full((batch_size, 1), V_MIN, device=device), reward + (V_MIN + atom_tens * DELTA_Z) * gamma)
#     tz_j = torch.minimum(torch.full((batch_size, 1), V_MAX, device=device), tz_j)
#     b_j = (tz_j - V_MIN) / DELTA_Z
    
def distr_projection(next_distr_v, rewards_v, dones_mask_t, gamma, device="cpu"):
    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    for atom in range(N_ATOMS):
        tz_j = np.minimum(V_MAX, np.maximum(V_MIN, rewards + (V_MIN + atom * DELTA_Z) * gamma))
        b_j = (tz_j - V_MIN) / DELTA_Z
        b_j = b_j.reshape(batch_size)

        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l

        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(V_MAX, np.maximum(V_MIN, rewards[dones_mask]))
        b_j = (tz_j - V_MIN) / DELTA_Z
        b_j = b_j.reshape(-1)

        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return torch.FloatTensor(proj_distr).to(device)

def update_critic(critic, target_critic, target_actor, crt_opt, buf):
    s_i, act, r, s_ii, is_done = buf

    critic.train()
    crt_opt.zero_grad()

    q_i = critic(s_i, act)
    a_ii = target_actor(s_ii)
    q_ii = F.softmax(target_critic(s_ii, a_ii), dim=1)

    prob_dist_v = distr_projection(q_ii, r, is_done, gamma=GAMMA**REWRAD_STEP, device=device)
    prob_dis_v = -F.log_softmax(q_i, dim=1) * prob_dist_v

    loss = prob_dis_v.sum(dim=1).mean()
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
    loss = critic(s_i, act_y_hat)
    loss = -critic.distr_to_q(loss)
    loss = loss.mean()
    
    loss.backward()
    act_opt.step()
    freeze_model(critic, is_freeze=False)
    return loss.item()


ENV_ID = 'MinitaurBulletEnv-v0'
RENDER = False

EPOCH = 1000
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
WEIGHT_DECAY = 0.001


BUFFER_SIZE = 100000
REPLAY_INITIAL = 10000


LAMBDA = 0.01
GAMMA = 0.99 #discount factor
TAU = 0.001


ACT_SIZE = 8

N_ATOMS = 51
V_MIN, V_MAX = -10., 10.
REWRAD_STEP = 5
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)

device = 'cuda'

his_r, his_act, his_cri = list(), list(), list()

def save_his(his_r, his_act, his_cri):
    plt.figure(figsize=(12, 5))
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    plt.subplot(1, 3, 1)
    plt.title(f'<Rward>')
    plt.plot(his_r, color='black', label='total')
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.title(f'<Actor Loss>')
    plt.plot(his_act, color='black', label='total')
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.title(f'<Critic Loss>')
    plt.plot(his_cri, color='black', label='total')
    plt.grid()

    plt.savefig('./his.png')

if __name__ == '__main__':
    spec = gym.envs.registry.spec(ENV_ID)
    spec._kwargs['render'] = RENDER
    env = gym.make(ENV_ID)

    buffer = replayBuffer.Buffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, device=device)

    actor = models.Actor(input_shape=28, output_shape=ACT_SIZE).to(device)
    critic = models.Critic(state_shape=28, act_size=ACT_SIZE, n_atoms=N_ATOMS, v_min=V_MIN, v_max=V_MAX).to(device)
    tg_actor, tg_critic = deepcopy(actor).to(device), deepcopy(critic).to(device)

    # act_opt = optim.SGD(actor.parameters(), lr=LEARNING_RATE, weight_decay=1e-6, momentum=0.9)
    # crt_opt = optim.SGD(critic.parameters(), lr=LEARNING_RATE, weight_decay=1e-6, momentum=0.9)

    act_opt = optim.Adam(actor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    crt_opt = optim.Adam(critic.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    ep = 0
    while True:
        LAMBDA = np.random.rand()
        ep += 1

        #collect buffer++++++++++++++++++++++++++
        time_step, rewards = replayOneEpisod(env, actor, buffer, action_size=ACT_SIZE, device=device)
        #End+++++++++++++++++++++++++++++++++++++

        if buffer.size() < REPLAY_INITIAL:
            continue

        #train model+++++++++++++++++++++++++++++
        buffer.update()
        buf = buffer.get_batch()
        
        cri_loss = update_critic(critic, tg_critic, tg_actor, crt_opt, buf)
        act_loss = update_actor(actor, critic, act_opt, buf)


        soft_update(actor, tg_actor, T=TAU)
        soft_update(critic, tg_critic, T=TAU)
        #End+++++++++++++++++++++++++++++++++++++

        print(f'({ep} / ∞) Reward: {rewards:.3f}, times: {time_step}, Cri: {cri_loss:.3f}, Act: {act_loss:.3f}')

        his_r.append(rewards)
        his_act.append(act_loss)
        his_cri.append(cri_loss)

        #Test++++++++++++++++++++++++++++++++++++
        if ep % 10 == 0:
            test_time_step, test_rewards = replayOneEpisod(env, actor, buffer=None, action_size=None, device=device)
            print(f'\nTEST Reward: {test_rewards:.3f}, times: {test_time_step}')

            save_his(his_r, his_act, his_cri)
        #End+++++++++++++++++++++++++++++++++++++
        print('=' * 100)


    env.close()
    