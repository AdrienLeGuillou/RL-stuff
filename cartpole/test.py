import numpy as np
import gym
import os
os.chdir('/home/jovyan/work/RL-stuff/cartpole')

from qmodel import Qmodel
from agent_model import Agent


env = gym.make('CartPole-v0')
qmod = Qmodel()
bot = Agent(env, Qmod=qmod)

a = env.action_space.sample()
s = env.observation_space.sample()

np.mean(bot.train(100000))

np.mean(bot.play(20000))

bot = Agent(env, Qmod=qmod, eps=0.01, reset=False)

np.mean(bot.train(100000))

np.mean(bot.play(20000))

import random

for _ in range(10):
    print(random.randrange(2))

for _ in range(10):
    print(np.random.randint(2))
