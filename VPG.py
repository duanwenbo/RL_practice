import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gym
import numpy as np
import wandb


# create policy network, to find the optimal policy
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


# create the state-value prediction net, to find the max v value when evaluating advantage func
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


v_net = V_net()
policy_net = Policy_net()


def advantage_value(rewards, state_values, gamma):  # state_value : tensor
    #  advantage = discounted rewards - estimate v value
    # 1. discounted rewards
    # input:
    #         vector x,
    #         [x0,
    #          x1,
    #          x2]
    #     output:
    #         [x0 + discount * x1 + discount^2 * x2,
    #          x1 + discount * x2,
    #          x2]

    discounted_rewards = np.zeros_like(rewards)
    addition = 0
    for i in reversed(range(0, len(rewards))):
        discounted_rewards[i] = rewards[i] + addition
        addition = discounted_rewards[i] * gamma
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

    advantages = discounted_rewards - state_values
    return advantages, discounted_rewards


def action_distribution(state):
    prob = policy_net(state).detach()
    return Categorical(prob)


env = gym.make('CartPole-v0')
LEARNING_RATE = 0.0005
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
v_optimizer = torch.optim.Adam(v_net.parameters(), lr=LEARNING_RATE)
v_loss_func = nn.MSELoss()
episodes_num = 1000
gamma = 0.98
for i in range(episodes_num):
    rewards, states, actions, log_pro = [], [], [], []
    state_values = torch.tensor([], dtype=torch.float32)
    state = env.reset()
    while True:
        action_distr = action_distribution(torch.as_tensor(state, dtype=torch.float32))
        action = action_distr.sample()
        next_state, reward, done, _ = env.step(action.item())

        # record trajectories
        rewards.append(reward)
        states.append(state)
        log_pro.append(action_distr.log_prob(action).item())
        actions.append(action)
        state_values = torch.cat((state_values, v_net(torch.as_tensor(state, dtype=torch.float32))), 0)  # append new predicted v values to the tensor array

        state = next_state

        if done:
            break

    # training the network
    advantage_values, discounted_rewards = advantage_value(rewards, state_values, gamma)
    policy_loss = - torch.sum(torch.mul(torch.tensor(log_pro), advantage_values))
    policy_loss.backward(retain_graph=True)
    policy_optimizer.step()
    # refit v function
    v_optimizer.zero_grad()
    v_loss = v_loss_func(discounted_rewards, state_values)
    v_loss.backward()
    v_optimizer.step()

    print("########################数据验证##################################")
    print("#################################################################")


