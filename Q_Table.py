import numpy as np
import matplotlib.pyplot as plt
import gym


class Agent:
    def __init__(self):
        self.alpha = 0.05  # learning rate
        self.gamma = 0.95  # discount rate
        self.epsilon_max = 1.0  # initial epsilon value
        self.epsilon_decay = 0.999
        self.episode_number = 10000
        self.total_reward = []
        self.total_step_length = []

    def q_learning_train(self, env):
        q_table = np.zeros((env.observation_space.n, env.action_space.n))  # initialize the q table
        epsilon = self.epsilon_max  # start epsilon rate from the highest

        for episode in range(self.episode_number):
            state = env.reset()
            episode_reward = 0
            step_length = 0
            while True:
                # choose action
                if np.random.uniform(0, 1) <= epsilon:
                    action = env.action_space.sample()  # exploration
                else:
                    action = np.argmax(q_table[state])  # exploitation

                # interact with the environment
                next_state, current_reward, done, _ = env.step(action)

                # update Q value & next state & rest
                q_table[state][action] += self.alpha * (
                        current_reward + self.gamma * np.max(q_table[next_state]) - q_table[state][
                    action])
                state = next_state
                episode_reward += current_reward
                step_length += 1

                # when a single episode is over, record results
                if done:
                    self.total_reward.append(episode_reward)
                    self.total_step_length.append(step_length)
                    print("Episode:{}  Reward:{}  Step Length:{}".format(episode, episode_reward, step_length))
                    break

            #  move action choosing strategy from exploration to exploitation as experience increases.
            epsilon *= self.epsilon_decay

        self.plot_result()
        return q_table

    def plot_result(self):
        def smooth_data(source, smooth_over):
            data = []
            for ii in range(smooth_over, len(source)):
                data.append(np.mean(source[ii - smooth_over:ii]))
            return data

        plt.figure(1)
        plt.plot(smooth_data(self.total_reward, 100))
        plt.ylabel("reward")
        plt.figure(2)
        plt.plot(smooth_data(self.total_step_length, 100))
        plt.ylabel("step length")
        plt.show()


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0", is_slippery=False)
    q_table = Agent().q_learning_train(env)
    print("########final q table#############\n{}".format(q_table))
    # test the final result in the game
    state = env.reset()
    while True:
        env.render()
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break

