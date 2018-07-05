import numpy as np
import gym
import os
os.chdir('/home/jovyan/work/RL-stuff/cartpole')

env = gym.make('CartPole-v0')

a = env.action_space.sample()
s = env.observation_space.sample()

env.action_space.n
env.observation_space.shape[0]
