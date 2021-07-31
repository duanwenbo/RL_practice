# Vanilla Policy Gradient with Openai Gym MountainCar-v0 environment
# Author: Bobby
# Time: 29-07-2021

import gym
import torch
import numpy as np
import wandb
from torch.distributions import Categorical
import copy

env = gym.make("MountainCar-v0")

# establish policy network
policy_net = torch.nn.Sequential(
    torch.nn.Linear(env.observation_space.shape[0], 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, env.action_space.n),
    torch.nn.Softmax()
)

target_net = torch.nn.Sequential(
    torch.nn.Linear(env.observation_space.shape[0], 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1)
)


# verify model
def verification():
    ob = env.reset()
    ep_reward = 0
    for i in range(EPISODE_NUM):
        action = Categorical(policy_net(torch.tensor(ob, dtype=torch.float32)).detach()).sample().item()
        ob_, reward, done, _ =env.step(action)
        ep_reward += reward
        ob = ob_
        if done:
            break
    return ep_reward


# hyper parameters
EPISODE_NUM = 10000
GAMMA = 0.98
LEARNING_RATE = 0.0005

# initialization
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
target_optimizer = torch.optim.Adam(target_net.parameters(), lr=LEARNING_RATE)
target_loss_func = torch.nn.MSELoss(reduction='sum')

# main loop
for ii in range(EPISODE_NUM):
    # initialization of recording trajectory
    actions = []
    states = []
    rewards = []
    state_values = []

    modified_rewards = 0

    state = env.reset()
    while True:
        action = Categorical(policy_net(torch.tensor(state, dtype=torch.float32)).detach()).sample().item()
        next_state, reward, done, _ = env.step(action)

        position, velocity = next_state
        modified_reward = (position + 0.5) ** 2
        modified_rewards += modified_reward


        # record trajectory
        state_value = target_net(torch.tensor(state, dtype=torch.float32)).detach().item()
        actions.append(action)
        states.append(state)
        rewards.append(modified_reward)
        state_values.append(state_value)

        # update state
        state = next_state
        if done:
            break
    # once one episode is done
    """
    optimize policy net
    loss = E[log_pi * Advantage]
    Advantage = E[discounted accumulation(TD difference)]
    """
    # compute log probabilities
    log_prob = Categorical(policy_net(torch.tensor(states, dtype=torch.float32))).log_prob(torch.tensor(actions, dtype=torch.float32))

    # compute advantage, using generalize advantage estimate there
    # convert from list to np.array for computation
    rewards, state_values = np.array(rewards), np.array(state_values)
    # delta = R(s,a) + gamma * max_V(s') - V(s)
    delta = rewards[:-1] + GAMMA * state_values[1:] - rewards[:-1]  # note one unit shorter than the original list
    # compute discounted accumulation
    for i in reversed(range(len(delta))):
        delta[i] = delta[i] + GAMMA*0.97*delta[i+1] if i+1 < len(delta) else delta[i]
    delta = torch.as_tensor(delta, dtype=torch.float32)

    # gradient ascent
    policy_loss = - torch.sum(torch.mul(log_prob[:-1], delta))
    # start optimization
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    """
    optimize target net
    loss = MSE(reward2go, state values)
    reward2go = discounted accumulation[:-1] ??
    """
    rtg = copy.copy(rewards)[:-1]
    for i in reversed(range(len(rtg))):
        rtg[i] = rtg[i] + GAMMA * rtg[i+1] if i+1 < len(rtg) else rtg[i]

    estimate_state_values = target_net(torch.tensor(states[:-1], dtype=torch.float32))
    target_loss = target_loss_func(torch.tensor(rtg, dtype=torch.float32), estimate_state_values.squeeze())
    # start optimization
    target_optimizer.zero_grad()
    target_loss.backward(retain_graph=True)
    target_optimizer.step()

    # verify
    result = verification()
    print("episode:{}, reward:{}, modified_reward:{}".format(ii, result, modified_rewards))




