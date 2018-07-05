import numpy as np
import gym
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import os
os.chdir('/home/jovyan/work/RL-stuff/cartpole')


env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, 'tmp/cartpole-experiment-1', force=True)
observation = env.reset()
for t in range(100):
    #env.render()
    #print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        #observation = env.reset()
        print("Episode finished after {} timesteps".format(t+1))
        break

env.close()

class Agent:
    def __init__(self, env, Qmod, algo="sarsa1", lr=0.1, dr=0.9, eps=0.1):
        """
        env = openAI gym env
        Qmod = Q(s, a) estimator
        lr = learning rate
        dr = discount rate
        eps = epsilon for the e-greedy policy
        """
        self.env = env
        self.lr = lr
        self.dr = dr
        self.eps = eps
        self.A = np.arange(2)
        self.algo = algo
        self.Qmod = Qmod
        self._algo(algo)
        self._reset()


    def _reset(self):
        pass


    def _algo(self, algo):
        if algo == "sarsa1":
            self._ep_train = self._train_sarsa1


    def _pick_action(self, s, greedy=False):
        Q = []
        for a in self.A:
            x = self._x_from_s_a(s, a)
            Q.append(self._q_from_x(x))

        if greedy:
            return self.A[np.argmax(Q)]
        else:
            p = np.random.random()
            if p <= self.eps:
                return np.random.choice(self.A)
            else:
                return self.A[np.argmax(Q)]


    def _x_from_s_a(self, s, a):
        """
        s is an 1d-array, a state returned by cartpole-v0
        a is a scalar, 0 or 1
        """
        return np.concatenate((s, np.array([a])))
        # return np.concatenate((s, np.array([a]),
        #                        np.array([a])**3,
        #                        np.array([1])))
        #return np.concatenate((s, np.array([a]), np.array([1])))
        #return np.concatenate((s[np.array([0, 2])], np.array([a])))


    def _q_from_x(self, x):
        #return self.W @ x
        return self.Qmod.predict(x.reshape(1,-1))


    def _update_mod(self, x, y):
        self.Qmod.fit(x.reshape(1,-1), y, verbose=False)


    def play(self, n=-1, greedy=False, display=False):
        if n < 0:
            n = self.env.target * 2
        l_steps = []
        while True:
            steps = 0

            s = self.env.reset()
            a = self._pick_action(s)
            done = False

            while not done and n > 0:
                s, r, done, info = self.env.step(a)
                a = self._pick_action(s, greedy)
                steps += 1
                n -= 1

            if done:
                l_steps.append(steps)
                if display:
                   self.env.render()
            else:
                break

        return l_steps


    def train(self, n=-1):
        if n < 0:
            n = self.env.target * 10000
        l_steps = []
        while True:
            steps, done = self._ep_train(n)
            n -= steps

            if done:
                l_steps.append(steps)
            else:
                break

        return l_steps


    def _train_sarsa1(self, n):
        done = False
        steps = 0

        s = self.env.reset()
        a = self._pick_action(s)

        while not done and n > 0:
            s_, r, done, info = self.env.step(a)

            a_ = self._pick_action(s_)

            x = self._x_from_s_a(s, a)
            x_ = self._x_from_s_a(s_, a_)
            y = r + self.dr * self._q_from_x(x_)

            self._update_mod(x, y)

            s, a = s_, a_

            steps += 1
            n -= 1

        return steps, done


mod_q = Sequential()
mod_q.add(Dense(16, activation='relu', input_dim=5))
mod_q.add(Dense(16, activation='relu'))
mod_q.add(Dense(1, activation='linear'))

mod_q.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])

env = gym.make('CartPole-v0')
bot = Agent(env, Qmod=mod_q, eps=0.1)

bot.train(100000)

for _ in range(20):
    print(np.mean(bot.train(200)))

for _ in range(20):
    print(np.mean(bot.play(200)))
