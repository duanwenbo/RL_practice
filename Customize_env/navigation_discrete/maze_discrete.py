from gym import Env
import gym
from gym.envs.registration import make
from gym.spaces import Discrete, Box
import numpy as np
import random
from scipy.interpolate import interp1d
import math
import torch
import torch.nn.functional as F
from torch import tensor



class Maze(Env):
    metadata = {"render.modes":["console"]}
    def __init__(self) -> None:
        super(Maze, self).__init__()
        self.action_space = Discrete(9)
        self.observation_space = Box(low=np.array([0,0]), high=np.array([10,10]))
        self.state = np.array([9.0,9.0])
        self.episode_length = 100
        self.action_dict = {"0":(0,0.1),"1":(0.1,0.1),"2":(0.1,0),"3":(0.1,-0.1),"4":(0,-0.1),"5":(-0.1,-0.1),"6":(-0.1,0),"7":(-0.1,0.1),"8":(0.0,0.0)}
        # self.goal = (random.uniform(0,10), random.uniform(0,10))
        self.goal = (0.1, 0.5) # for model test
        self.map = interp1d([-1,1], [0,360])

    def step(self, action):
        self.episode_length -= 1
        # action = self.map(np.tanh(action))
        # x_increment, y_increment = 0.3 * math.cos(math.radians(action)), 0.3 * math.sin(math.radians(action))
        self.state = (self.state[0]+ self.action_dict[str(action)][0], self.state[1]+self.action_dict[str(action)][1])
        distance = ((self.state[0] - self.goal[0])**2 + (self.state[1] - self.goal[1])**2)**0.5
        amplify_factor= max([0.008, self.episode_length / 100]) # normalized amplified factor
        reward = (-distance) / amplify_factor
        if self.episode_length <=0:
            done = True
        elif distance <= 0.05:
            done = True
            reward += 500
        else:
            done = False
        return np.array(self.state), reward, done, distance

    def reset(self):
        self.state =  np.array([9.0,9.0])
        self.episode_length = 100
        return self.state

    def render(self):
        pass
        

if __name__ == "__main__":
    maze = Maze()
    print(maze.step(60))
