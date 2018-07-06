import os
os.chdir('/home/jovyan/work/RL-stuff/udemy-DRL')

import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as k
from keras.layers import Dense, Dropout
from keras.models import Sequential


class Toy_NN:
    def __init__(self, input_shape):
        """
        input_shape: tupple containing the shape (4,) in the case of CartPole
        """
        print('Hello Keras NN')

        model = Sequential()

        model.add(Dense(200, activation='relu',
                        input_shape=input_shape))
        # model.add(Dropout(0.5))
        model.add(Dense(200, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(2, activation='linear'))

        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mse'])

        self.model = model

    def fit(self, X, Y):
        self.model.fit(X, Y, epochs=1, verbose=0)

    def predict(self, X):
        return self.model.predict(X, verbose=0)


class IO_transformer:
    def __init__(self, s_shape, a_size):
        self.s_shape = s_shape
        self.a_size = a_size

    def s_to_x(self, observation):
        observation = observation.reshape(((-1,) + self.s_shape))

        return observation

    def y_to_actions(self, y):
        return y


class Agent:
    def __init__(self, env):
        self.env = env

        obs_sample = env.observation_space.sample()
        s_shape = obs_sample.shape
        a_size = env.action_space.n
        self.io_transformer = IO_transformer(s_shape, a_size)
        self.make_model(s_shape)

    def make_model(self, s_shape):
        self.model = Toy_NN(s_shape)

    def predict(self, s):
        X = self.io_transformer.s_to_x(s)
        return self.model.predict(X)

    def update(self, s, a, G):
        X = self.io_transformer.s_to_x(s)
        Y = self.model.predict(X)
        Y[0, a] = G
        self.model.fit(X, Y)

    def pick_action(self, s, eps):
        if np.random.random() < eps:
            a = self.env.action_space.sample()
        else:
            y = self.predict(s)
            actions = self.io_transformer.y_to_actions(y)
            a = np.argmax(actions, axis=1)[0]

        return a


def play_one(bot, eps, gamma):
    s = bot.env.reset()
    done = False
    total_reward = 0
    iters = 0

    while not done and iters < 2000:
        a = bot.pick_action(s, eps)
        s_, r, done, _ = bot.env.step(a)

        if done:
            r = -200

        G = r + gamma*np.max(bot.predict(s_))
        bot.update(s, a, G)
        s = s_

        if not done:
            total_reward += r
        iters += 1

    return total_reward


def plot_running_avg(total_rewards):
    n = len(total_rewards)
    running_avg = np.cumsum(total_rewards) / (np.array(range(n)) + 1)

    plt.plot(running_avg)
    plt.title('Running Average')
    plt.show()


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    bot = Agent(env)
    gamma = 0.99

    N = 200
    total_rewards = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        total_rewards[n] = play_one(bot, eps, gamma)
        print(f'Episode {n+1}/{N}, total_reward = {total_rewards[n]}')


    plt.plot(total_rewards)
    plt.title("Rewards")
    plot_running_avg(total_rewards)

    env = gym.wrappers.Monitor(env, 'monitor/cartpole_nn', force=True)
    bot.env = env
    play_one(bot, 0, gamma)

    env.close()
