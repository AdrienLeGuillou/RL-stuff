from collections import deque
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import gym
import random
import os
os.chdir('/home/jovyan/work/RL-stuff/cartpole')

class DQNAgent:
    def __init__(self, env, memsize=2000, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.005, learning_rate=0.001):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=memsize)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.seen_done = False


    def _build_model(self):
        model = Sequential()

        model.add(Dense(36, activation='relu', input_dim=self.state_size))
        model.add(Dense(36, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state, greedy=False):
        if not greedy and np.random.randn() <= self.epsilon:
            action =  np.random.randint(self.action_size)
        else:
            act_values = self.model.predict(state)
            action =  np.argmax(act_values[0])

        return action


    def replay(self, batch_size):
        batch_size = np.minimum(batch_size, len(self.memory))

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
                self.seen_done = True
            else:
                target = reward + self.gamma \
                         * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min and self.seen_done:
            self.epsilon *= (1 - self.epsilon_decay)


    def play(self, episodes=1, max_t=500,
             record=False, record_path=''):

        if record:
            if record_path is '':
                record_path = 'tmp/' + self.env.spec.id
            self.env = gym.wrappers.Monitor(self.env, record_path, force=True)

        times = []
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, (1, -1))

            for time_t in range(max_t):
                action = self.act(state, greedy=True)

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, -1))
                state = next_state

                if done:
                    self.seen_done = True
                    break

            times.append(time_t)

        if record:
            self.env.close()
            self.env = self.env.unwrapped

        return times


    def train(self, episodes=1, max_t=500, replay_size=32, verbose=False):
        times = []
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, (1, -1))

            for time_t in range(max_t):

                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, -1))

                self.remember(state, action, reward, next_state, done)

                if done:
                    target = reward
                    self.seen_done = True
                else:
                    target = reward + self.gamma \
                             * np.amax(self.model.predict(next_state)[0])

                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)

                state = next_state

                if done:
                    self.seen_done = True
                    break

            if verbose:
                print(f'episode {e + 1} out of {episodes}: {time_t} steps. Done was {done}')
            times.append(time_t)

            self.replay(replay_size)

        return times



env = gym.make('CartPole-v0')
agent = DQNAgent(env)

agent.train(50)
agent.play(record=True)

env = gym.make('Acrobot-v1')
agent = DQNAgent(env)

agent.train(500)
agent.play(record=True)
