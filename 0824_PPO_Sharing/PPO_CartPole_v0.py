import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gym
import numpy as np
import wandb

env = gym.make("CartPole-v0")

wandb.init("CartPole_v0_0824")



# create policy network
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


# create state_value network
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


# choose_action based on current policy, to interact with the environment on each step
def choose_action(state):
    """
    :param state: array
    :return: init, a single action
    """
    state = torch.as_tensor(state, dtype=torch.float32)
    dist = Categorical(current_policy_net(state).detach())
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob


"""
loss = Et[ min(Rt*At,  clip(Rt, 1-epsilon, 1+ epsilon)*At) - lamda*(state_value loss) ]
"""


def compute_advantage(state_values, rewards, gamma, gae_lambda):
    """
    :param state_values: list, already estimated from env
    :param rewards: list from env
    :param gamma: float
    :param gae_lambda: float
    :return: float tensor
    """
    # compute delta
    # note the advantage length is one unit smaller
    rewards, state_values = np.array(rewards), np.array(state_values)  # convert from list to array for calculation
    delta = rewards[:-1] + gamma * state_values[1:] - state_values[:-1]

    # compute discounted accumulation
    for i in reversed(range(len(delta))):
        delta[i] = delta[i] + gamma * gae_lambda * delta[i + 1] if i + 1 < len(delta) else delta[i]
    gae_advantage = torch.as_tensor(delta, dtype=torch.float32)
    return gae_advantage


# probability ratio, used to compute loss at the end of each episode
def compute_prob_ratio(states, actions, old_probs):
    """
    :param states: list from env
    :param actions:  list from env
    :return: float tensor
    """
    states = torch.tensor(states[:-1], dtype=torch.float32)
    actions = torch.tensor(actions[:-1], dtype=torch.float32)
    new_probs = Categorical(current_policy_net(states)).log_prob(actions)
    old_probs = torch.tensor(old_probs[:-1], dtype=torch.float32)
    prob_ratio = new_probs.exp() / old_probs.exp()
    return prob_ratio


"""
v_loss = (discounted_return - estimate state_values)^2
"""


def compute_v_loss(rewards, gamma, states):
    """
    :param rewards: list from env
    :param gamma: float
    :param states: list from env
    :return: tensor, for computing final loss
    """
    # deal with length first, in order to match the length of advantage value
    states, rewards = states[:-1], rewards[:-1]

    # discounted return
    discounted_return = np.zeros_like(rewards)
    addition = 0
    for i in reversed(range(len(rewards))):
        discounted_return[i] = rewards[i] + addition
        addition = discounted_return[i] * gamma
    discounted_return = torch.as_tensor(discounted_return, dtype=torch.float32)

    # estimate_v
    estimate_v = v_net(torch.tensor(states, dtype=torch.float32)).squeeze()

    # v_loss
    v_loss = (discounted_return - estimate_v) ** 2
    v_loss = v_loss.mean()
    return v_loss


def verify():
    ep_reward = 0
    state = env.reset()
    while True:
        action, _ = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        state = next_state
        if done:
            break
    return ep_reward


# hyperparameters
GAMMA = 0.99
EPSILON = 0.2
GEA_LAMBDA = 0.97
LEARNING_RATE = 0.0005
EPISODE_NUM = 620
BATCH_SIZE = 5
EPOCH = 4
TRAJECTORY_LENGTH = 20


# initialization
current_policy_net = Policy_net()
old_policy_net = Policy_net()
v_net = V_net()
current_policy_optimizer = torch.optim.Adam(current_policy_net.parameters(), lr=LEARNING_RATE)
v_net_optimizer = torch.optim.Adam(v_net.parameters(), lr=LEARNING_RATE)
current_policy_net.load_state_dict(old_policy_net.state_dict())  # synchronous two nets before starting

# main loop
for i in range(EPISODE_NUM):
    # record trajectories, reset at the beginning of each episode
    states = []
    rewards = []
    actions = []
    next_states = []
    state_values = []
    old_log_probs = []
    state = env.reset()  # initialize state, a 1x4 array
    done = False
    while not done:
        action, log_prob = choose_action(state)  # use current policy net
        next_state, reward, done, _ = env.step(action)
        # start recording
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        next_states.append(next_state)
        old_log_probs.append(log_prob)
        state_values.append(v_net(
            torch.as_tensor(state, dtype=torch.float32)).detach().item())
        state = next_state
       

    # pass the parameters from the current net to the old net before optimization
    for _ in range(EPOCH):

        # once an episode finished, compute objective function
        prob_ratio = compute_prob_ratio(states, actions, old_log_probs)
        advantage = compute_advantage(state_values, rewards, GAMMA, GEA_LAMBDA)
        clip_loss = - torch.min(prob_ratio * advantage, torch.clamp(prob_ratio, 1 - EPSILON, 1 + EPSILON) * advantage).mean()
        v_loss = compute_v_loss(rewards, GAMMA, states)
        # total_Loss = - clip_loss + C1 * v_loss

        # update network
        current_policy_optimizer.zero_grad()
        clip_loss.backward()
        current_policy_optimizer.step()

        v_net_optimizer.zero_grad()
        v_loss.backward()
        v_net_optimizer.step()

    # verify
    result = verify()
    print("episode:{}    rewards:{}".format(i, result))
    wandb.log({"epoch": i, "reward": result})