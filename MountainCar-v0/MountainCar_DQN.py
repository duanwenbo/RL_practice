import random
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import wandb

# wandb.init(project="MountainCar-v0-DQN")
env = gym.make("MountainCar-v0")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, input=env.observation_space.shape[0], hiddden=64, output=env.action_space.n):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input, hiddden)
        self.fc2 = nn.Linear(hiddden, hiddden)
        self.fc3 = nn.Linear(hiddden, output)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)
        return x


def verify():
    ob = env.reset()
    total_reward = 0
    while True:
        env.render()
        ob_tensor = torch.as_tensor(ob, dtype=torch.float32).to(device)
        action = policy_net(ob_tensor).argmax(0).detach().item()
        ob_, reward, done, _ = env.step(action)
        total_reward += reward
        ob = ob_
        if done:
            break
    return total_reward


"""
DQN: experience replay
fixed Q type
loss =  MSE( Q(s,a) + gamma * max_a Q(s',a'),   Q(s,a))
"""

# initialize hyperparameters
EPISODE_NUM = 10000
BATCH_SIZE = 50
LEARNING_RATE = 0.0001
GAMMA = 0.95
MEMORY_SIZE = 10000
EPSILON = 1.0
SYNC_STEP = 10
EPSILON_STEP = 0.000995
step = 10

policy_net = Net().to(device)
target_net = Net().to(device)
policy_net.load_state_dict(target_net.state_dict())
policy_net_optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
policy_nett_loss = nn.MSELoss().to(device)

experience_pool = deque([], maxlen=MEMORY_SIZE)
# main loop
for i in range(EPISODE_NUM):
    state = env.reset()
    modified_rewards = 0
    while True:
        experience = {}
        #  epsilon greedy
        if np.random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample()
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
            action = policy_net(state_tensor).argmax(0).detach().item()

        # interact with the environment, collect experience
        state_, reward, done, _ = env.step(action)

        # modified the reward system
        position, velocity = state_
        modified_reward = (position + 0.5) ** 2
        modified_rewards += modified_reward

        experience["state_"] = state_
        experience["reward"] = modified_reward
        experience["action"] = action
        experience["state"] = state
        experience["done"] = done
        experience_pool.append(experience)
        state = state_

        # learn from experience pool
        if len(experience_pool) >= BATCH_SIZE:
            """
            random sample a batch size of experience to learn
            [{a=1,b=2,c=3}, {a=1,b=2,c=3}, {a=1,b=2,c=3}]
            """
            training_batch = random.sample(experience_pool, BATCH_SIZE)
            batch_state_ = torch.as_tensor([item["state_"] for item in training_batch], dtype=torch.float32).to(device)
            batch_state = torch.tensor([item["state"] for item in training_batch], dtype=torch.float32).to(device)
            batch_reward = torch.tensor([item["reward"] for item in training_batch], dtype=torch.float32).to(device)
            batch_action = torch.tensor([item["action"] for item in training_batch], dtype=torch.int64).to(device)
            done_list = [item["done"] for item in training_batch]  # extract info from experience dict
            done_list = [1 if i is False else 0 for i in
                         done_list]  # change bool into int, for cpmputing the non final mask later
            batch_done = torch.tensor(done_list).to(device)

            # Q(s,a)
            batch_q_current = policy_net(batch_state).gather(1, batch_action.unsqueeze(1))

            # Q^*(s,a) = R(s,a) + gamma * Q(s',a')
            batch_q_next = target_net(batch_state_).max(1)[0].unsqueeze(1).detach().to(device)
            non_final_mask = batch_q_next * batch_done.unsqueeze(1)  # if this state is the final state of current episode, remains 0
            batch_q_optimal = batch_reward.unsqueeze(1) + GAMMA * non_final_mask

            # optimize
            policy_net_optimizer.zero_grad()
            loss = policy_nett_loss(batch_q_optimal, batch_q_current)

            loss.backward()
            policy_net_optimizer.step()

        if done:

            # synchronous target net and policy net
            step += 1
            if step % SYNC_STEP == 0:
                target_net.load_state_dict(policy_net.state_dict())

            ## EPSILON = max((EPSILON - EPSILON_STEP), 0.005) for CartPole
            EPSILON *= 0.995
            result = verify()
            # move from exploration to exploitation
            # print("episode:{}, reward:{}, modified_reward:{}, epsilon:{}".format(i, result, modified_rewards, EPSILON))
            # wandb.log({"episode":i, "reward": result})
            break
