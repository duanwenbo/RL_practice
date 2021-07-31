import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import copy
import wandb

env = gym.make("MountainCar-v0")
wandb.init(project="MountainCar-v0-PPO")


class Policy_net(nn.Module):
    def __init__(self, input=env.observation_space.shape[0], hidden=64, output=env.action_space.n):
        super(Policy_net, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))  # for probability
        return x


class Target_net(nn.Module):
    def __init__(self, input=env.observation_space.shape[0], hidden=64, output=1):
        super(Target_net, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def choose_action(state):
    """
    :param state: np.array
    :return: init
    """
    state = torch.tensor(state, dtype=torch.float32)
    dist = Categorical(new_policy_net(state).detach())
    action = dist.sample().item()
    return action


def compute_advantage(rewards, state_values, gamma):
    """
    :param rewards: [float]
    :param state_values: [float]
    :return: tensor, generalized advantage estimate
    one unit shorter !!
    """
    rewards, state_values = np.array(rewards), np.array(state_values)
    delta = rewards[:-1] + gamma * state_values[1:] - state_values[:-1]
    for i in reversed(range(len(delta))):
        delta[i] = delta[i] + gamma * 0.97 * delta[i + 1] if i + 1 < len(delta) else delta[i]
    gae_advantage = torch.as_tensor(delta, dtype=torch.float32)
    return gae_advantage


def compute_prob_ratio(states, actions):
    """
    :param states: [np.array]
    :param actions: [init]
    :return: tensor
    one unit shorter
    """
    states = torch.tensor(states[:-1], dtype=torch.float32)
    actions = torch.tensor(actions[:-1], dtype=torch.float32)
    new_prob = Categorical(new_policy_net(states)).log_prob(actions)
    old_prob = Categorical(old_policy_net(states)).log_prob(actions)
    prob_ratio = new_prob.exp() / old_prob.exp()
    return prob_ratio


def compute_rtg(rewards, gamma):
    """
    :param rewards: [float]
    :return: tensor
    one unit shorter
    """
    rtg = copy.copy(rewards[:-1])
    for i in reversed(range(len(rtg))):
        rtg[i] = rtg[i] + gamma * rtg[i + 1] if i + 1 < len(rtg) else rtg[i]
    return torch.tensor(rtg, dtype=torch.float32)


def verify():
    ep_reward = 0
    state = env.reset()
    while True:
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        state = next_state
        if done:
            break
    return ep_reward


#
LEARNING_RATE = 0.0002
EPISODE_NUM = 20000
GAMMA = 0.98
EPSILON = 0.2
C1 = 0.5
C2 = 0.2

# initialization
new_policy_net = Policy_net()
old_policy_net = Policy_net()
target_net = Target_net()
policy_optimizer = torch.optim.Adam(new_policy_net.parameters(), lr=LEARNING_RATE)
target_optimizer = torch.optim.Adam(target_net.parameters(), lr=LEARNING_RATE)
target_loss_func = torch.nn.MSELoss()
new_policy_net.load_state_dict(old_policy_net.state_dict())

# main loop
for i in range(EPISODE_NUM):
    actions = []
    rewards = []
    state_values = []
    states = []
    ob = env.reset()
    modified_rewards = 0
    while True:
        action = choose_action(ob)
        ob_, reward, done, _ = env.step(action)

        position, velocity = ob_
        modified_reward = (position + 0.5) ** 2
        modified_rewards += modified_reward

        actions.append(action)
        rewards.append(modified_reward)
        states.append(ob)
        state_values.append(target_net(torch.tensor(ob, dtype=torch.float32)).detach().item())

        ob = ob_
        if done:
            break
    """
    E[min(prob_ratio*advantage, clip(prob_ratio)*advantage) - c_1* MSE(rtg, state values) + ..]
    """
    old_policy_net.load_state_dict(new_policy_net.state_dict())

    # compute clip loss first
    prob_ratio = compute_prob_ratio(states, actions)
    advantage = compute_advantage(rewards, state_values, GAMMA)
    clip_loss = torch.min(prob_ratio * advantage, torch.clamp(prob_ratio, 1 - EPSILON, 1 + EPSILON) * advantage).mean()

    # compute target loss
    rtg = compute_rtg(rewards, GAMMA)
    v_values = target_net(torch.tensor(states[:-1], dtype=torch.float32)).squeeze()
    target_loss = ((rtg - v_values) ** 2).mean()

    # compute entropy
    entropy = Categorical(new_policy_net(torch.tensor(states[:-1], dtype=torch.float32).detach())).entropy()
    batch_entropy = torch.mean(entropy)

    # total loss
    total_loss = - clip_loss + C1 * target_loss - C2 * batch_entropy

    # optimize network together
    policy_optimizer.zero_grad()
    target_optimizer.zero_grad()
    total_loss.backward()
    policy_optimizer.step()
    target_optimizer.step()

    # test
    result = verify()
    print("episode:{}    rewards:{}  modified_rewards:{}".format(i, result, modified_rewards))
    wandb.log({"episode": i, "reward": result, "modified rewards": modified_rewards})

