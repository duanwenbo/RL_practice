import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gym
import numpy as np
import wandb

env = gym.make("CartPole-v0")
wandb.init(project="CartPole_VPG")


"""
pay attention to tell the different usage of tensor detach() !!
advoid double float
"""


# policy network. to find the optimal policy and choose action
class Policy_net(nn.Module):
    def __init__(self, input=4, hidden=64, output=2):
        super(Policy_net, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))  # for probability
        return x


# state value estimate network, to find the optimal state value and choose the state value
class V_net(nn.Module):
    def __init__(self, input=4, hidden=64, output=1):
        super(V_net, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)
        return x


# choosing action based on state, using the policy network
def choose_action(state):
    """
    :param state: a 1x4 numpy array, which is returned from gym env
    :return: action: init (1 or 0), used as input to this gym environment
    """
    state = torch.as_tensor(state, dtype=torch.float32)  # convert from numpy to tensor
    probabilities = Categorical(policy_net(state).detach())  # prediction don't need to back propagate
    return probabilities.sample().item()


def compute_advantage(rewards, state_values, gamma):
    """
    :param rewards: list, contains reward value return by the environment
    :param state_values: list, contains expected v value evaluated by the v net
    :return: numpy array ---> tensor
    """
    # advantage func = discount_cumulative_sum(TD error)
    # TD error: R(s, a) + gamma * V(s') - V(s)
    rewards, state_values = np.array(rewards), np.array(state_values)  # convert from list to array for calculation
    td_error = rewards[:-1] + gamma * state_values[1:] - state_values[:-1]

    # discounted accumulation
    discounted_advantage = np.zeros_like(td_error)
    addition = 0
    for i in reversed(range(0, len(td_error))):
        discounted_advantage[i] = td_error[i] + addition
        addition = discounted_advantage[i] * gamma * 0.97
    return torch.tensor(discounted_advantage)


def compute_reward2go(rewards, gamma):
    """
    :param rewards: list, contains reward value return by the environment
    :return: numpy array ---> tensor,  while popping the last element
    """
    # discounted accumulation
    discounted_rewards = np.zeros_like(rewards)
    addition = 0
    for i in reversed(range(0, len(rewards))):
        discounted_rewards[i] = rewards[i] + addition
        addition = discounted_rewards[i] * gamma
    return torch.as_tensor(discounted_rewards[:-1], dtype=torch.float32)


def verify():
    # a separate func to test the training results of policy net
    rewards = 0
    state = env.reset()
    while True:
        action = choose_action(state)  # only for verification
        next_state, reward, done, _ = env.step(action)
        state = next_state
        rewards += reward
        if done:
            break
    return rewards


# hyper parameters
episode_num = 10000
gamma = 0.98
learning_rate = 0.0005

# initialization
policy_net = Policy_net()
v_net = V_net()
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
v_optimizer = torch.optim.Adam(v_net.parameters(), lr=learning_rate)
v_loss_func = nn.MSELoss(reduction='sum')  # ?

# main loop
for i in range(episode_num):
    # record trajectories, reset at the beginning of each episode
    states = []
    rewards = []
    actions = []
    next_states = []
    state_values = []
    state = env.reset()  # initialize state, a 1x4 array
    while True:
        # interact with the environment
        aciton = choose_action(state)
        next_state, reward, done, _ = env.step(aciton)
        # start recording
        states.append(state)
        rewards.append(reward)
        actions.append(aciton)
        next_states.append(next_state)
        state_values.append(v_net(torch.as_tensor(state, dtype=torch.float32)).detach().item())  # cut down back propagation, don't need gradient calcu later
        state = next_state
        if done:
            break
    # after a single episode is finished
    """
    optimize policy network
    loss = - log_probabilities * advantage func
    advantage func = discount_cumulative_sum(TD error) ??
    """
    states = torch.tensor(states[:-1], dtype=torch.float32)  # pop the last element since taking the last one as "next state"
    actions = torch.tensor(actions[:-1], dtype=torch.float32)

    policy_optimizer.zero_grad()
    log_probabilities = Categorical(policy_net(states)).log_prob(actions)  # tensor. steps size
    advantages = compute_advantage(rewards, state_values, gamma)  # tensor
    policy_loss = - torch.sum(torch.mul(log_probabilities, advantages))  # gradient ascent
    policy_loss.backward()
    policy_optimizer.step()

    """
    optimize state value network
    loss = MSE(reward2go v values, estimated v values)
    reward2go = discounted rewards[:-1] ??
    """
    v_optimizer.zero_grad()
    rewards2go = compute_reward2go(rewards, gamma)
    estimate_v = v_net(states).squeeze()
    v_loss = v_loss_func(rewards2go, estimate_v)
    v_loss.backward(retain_graph=True)  # ?
    v_optimizer.step()

    # verification
    result = verify()

    # output info
    print("episode:{}  reward:{}".format(i, result))
    wandb.log({"epoch": i, "reward": result})

