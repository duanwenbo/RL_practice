import random
import torch
import torch.nn as nn
import gym
import numpy as np
from collections import deque, namedtuple
import wandb


Net = torch.nn.Sequential(
    torch.nn.Linear(4, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 2)
)

# create an experience container
Experience = namedtuple("experience", ["state", "action", "reward", "next_state","done"])

# create the environment
env = gym.make("CartPole-v0")

device = torch.device("cpu")


class Agent:
    def __init__(self, epsilon_max, memory_size, batch_size, learning_rate, gamma, epsilon_decay, epsilon_min,
                 update_step, N_EPS):
        self.epsilon = epsilon_max
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_step = update_step
        self.memory = deque([], maxlen=memory_size)
        self.policy_net = Net
        self.target_net = Net
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
        self.step = 0
        self.loss = 0
        self.epsilon_step = (epsilon_max - epsilon_min) / N_EPS
        self.q_target = 0
        self.q = 0
        self.yes = 0

    def select_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample()
        else:
            # state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)  # convert to tensor type
            # values = self.policy_net(state_tensor)  # result = different possible actions with it's q value
            # _, max_action = torch.max(values,
            #                           1)  # returns maximum of each row with it's index, where index is the action
            # max_action = max_action.item()  # extract value from tensor
            state_tensor = torch.FloatTensor([state]).to(device)
            actions = self.policy_net(state_tensor).argmax(1).detach()
            max_action = actions.cpu().numpy()[0]
            return max_action

    def remember(self, experience):
        self.memory.append(experience)

    def sampling(self):
        return random.sample(self.memory, self.batch_size)

    def learn(self, experience):  # Deep Q learning with reward replay
        self.remember(experience)
        if len(self.memory) >= self.batch_size:
            self.yes = 1
            # extract training data from the experience pool
            training_batch = self.sampling()
            # sort data by type
            # example: experience(state=(1, 111, 11), action=(2, 222, 22))
            #training_batch = Experience(*zip(*training_batch))
            #print(training_batch)
            # extract specific type data, converting from list to tensor
            # states = torch.FloatTensor(list(training_batch.state))
            states = torch.FloatTensor([i.state for i in training_batch])
            # actions = torch.LongTensor(list(training_batch.action))
            actions = torch.LongTensor([i.action for i in training_batch])
            # rewards = torch.FloatTensor(list(training_batch.reward))
            rewards = torch.FloatTensor([i.reward for i in training_batch])
            # next_states = torch.FloatTensor(list(training_batch.next_state))
            next_states = torch.FloatTensor([i.next_state for i in training_batch])
            done_flags = [i.done for i in training_batch]
            # non_final_next_states = torch.FloatTensor([s for s in training_batch.next_state if s is not None])
            # Q(s, a)
            q_value = self.policy_net(states).gather(1, actions.unsqueeze(1))  # output as a column, find the largest q between actions from row 0 respectively
            self.q = q_value
            # maxQ(s', a')
            # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
            #                                         training_batch.next_state)), dtype=torch.bool)
            # q_next_max = torch.zeros(self.batch_size)  # initialize next_q column with zero
            # q_next_max[non_final_mask] = self.target_net(non_final_next_states).max(1)[
            #     0].detach()  # assign value if next state is not none, otherwise stay 0
            #
            # # Q*(s, a) = R(s, a) + y*maxQ(s', a')
            # q_optimal = rewards + self.gamma * q_next_max
            # self.q_target = q_optimal.unsqueeze(1)

            # 0719
            non_final_mask = torch.tensor(list(map(lambda s: s is not True, done_flags)), dtype=torch.float32).to(device)  #
            q_next_max = self.target_net(next_states).max(1)[0].detach()
            q_optimal = rewards + self.gamma * torch.mul(q_next_max, non_final_mask).to(device)
            q_optimal = torch.unsqueeze(q_optimal, 1)
            self.q_target = q_optimal


            self.optimizer.zero_grad()
            loss = self.loss_func(q_optimal, q_value)
            self.loss = loss
            loss.backward()
            self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max((self.epsilon - self.epsilon_step), self.epsilon_min)

    def update_target_net(self):
        if self.step % self.update_step == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step += 1

    def validate(self, env):
        ep_reward = 0
        state = env.reset()
        while True:
            state_tensor = torch.FloatTensor([state]).to(device)
            action = self.policy_net(state_tensor).argmax(1)
            action = action.cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if done:
                break
        return ep_reward




episodes_num = 10000
epsilon_max = 1.0
memory_size = 10000
batch_size = 50
learning_rate = 0.0001
gamma = 0.95
epsion_min = 0.005
N_steps = 1000
epsilon_decay = 0.9997
update_step = 10

wandb.init(project="CartPole")



episodes_reward = []

agent = Agent(epsilon_max, memory_size, batch_size, learning_rate, gamma, epsilon_decay, epsion_min, update_step, N_steps)
agent.update_target_net()
wandb.watch(agent.policy_net, log="all")
for i in range(episodes_num):
    state = env.reset()
    current_reward = 0
    while True:
        # env.render()
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        experience = Experience(state, action, reward, next_state, done)
        agent.learn(experience)
        state = next_state
        current_reward += reward
        loss = agent.loss
        # if agent.yes == 1:
        #     print("q_value")
        #     print(agent.q)
        #     print(len(agent.q))
        #     print("target q value")
        #     print(agent.q_target)
        #     print(len(agent.q_target))
        #     wandb.log({"epoch": i, "reward": current_reward, "Q_target": agent.q_target, "Q": agent.q})
        if done:
            ep_reward = agent.validate(env)
            wandb.log({"epoch": i, "reward": ep_reward, "epsilon": agent.epsilon})
            agent.update_epsilon()
            agent.update_target_net()
            print("episode:{}, reward:{}, epsilon{}".format(i, ep_reward, agent.epsilon))

            break
