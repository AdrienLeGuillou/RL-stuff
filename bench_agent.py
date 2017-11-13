from gridworld.gridworld import Gridworld
from agent.ep_agent import Agent
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import time

world = np.array([
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 9, 9, 1, 2, 2, 1, 0],
    [7, 0, 0, 9, 9, 1, 2, 8, 1, 0],
    [0, 0, 0, 9, 9, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
])

world_hard = np.array([
    [0, 1, 1, 9, 9, 9, 1, 1, 1, 0, 0, 0, 2, 2, 2, 9, 9, 0, 0, 0],
    [0, 1, 1, 9, 9, 9, 1, 1, 1, 0, 0, 0, 2, 2, 2, 9, 9, 0, 0, 0],
    [7, 1, 1, 9, 9, 9, 1, 1, 1, 0, 0, 0, 2, 2, 2, 9, 9, 0, 0, 0],
    [0, 1, 1, 9, 9, 9, 1, 1, 1, 0, 0, 0, 2, 2, 2, 9, 9, 0, 0, 0],
    [0, 1, 1, 9, 9, 9, 1, 1, 1, 9, 9, 0, 2, 2, 2, 9, 9, 0, 0, 0],
    [0, 1, 1, 9, 9, 9, 1, 1, 1, 9, 9, 0, 2, 2, 2, 9, 9, 0, 0, 0],
    [0, 1, 1, 9, 9, 9, 1, 1, 1, 9, 9, 0, 2, 2, 2, 9, 9, 0, 0, 0],
    [0, 1, 1, 9, 9, 9, 1, 1, 1, 9, 9, 0, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 1, 1, 2, 2, 2, 1, 1, 1, 9, 9, 0, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 1, 1, 2, 2, 2, 1, 1, 1, 9, 9, 0, 2, 2, 2, 0, 0, 0, 0, 0],
    [0, 1, 1, 2, 2, 2, 1, 1, 1, 9, 9, 0, 2, 2, 2, 9, 9, 0, 0, 0],
    [0, 1, 1, 2, 2, 2, 1, 1, 1, 9, 9, 0, 2, 2, 2, 9, 9, 0, 0, 0],
    [0, 1, 1, 2, 2, 2, 1, 1, 1, 9, 9, 0, 2, 2, 2, 9, 9, 0, 0, 0],
    [0, 1, 1, 2, 2, 2, 1, 1, 1, 9, 9, 0, 2, 2, 2, 9, 9, 0, 0, 0],
    [0, 1, 1, 2, 2, 2, 1, 1, 1, 9, 9, 0, 2, 2, 2, 9, 9, 8, 0, 0],
    [0, 1, 1, 2, 2, 2, 1, 1, 1, 9, 9, 0, 2, 2, 2, 9, 9, 0, 0, 0],
    [0, 1, 1, 2, 2, 2, 1, 1, 1, 9, 9, 0, 2, 2, 2, 9, 9, 0, 0, 0]
])

env = Gridworld(world_hard, (-1, 0), 75)

bot = Agent(env, algo="sarsa")

# ep_step = bot.train(2000)

# plt.plot(ep_step)
# plt.plot(savgol_filter(ep_step, 55, 1, mode='nearest'))
# plt.axhline(10 * env.target, c='red')
# plt.axhline(env.target, c='green')
# plt.yscale('log')
# # plt.xscale('log')
# plt.show()

# bot.play(n=5, greedy=True, display=True)

def bench_algo(bot, n_ep=100):
    start = time.time()
    steps = 0
    ep = 0
    greedy = False
    result = {
        'algo': bot.algo,
        'lr' : bot.lr,
        'tr' : bot.tr,
        'eps' : bot.eps
    }

    while True:
        train_steps = bot.train(n_ep)
        ep += n_ep
        steps += np.sum(train_steps)
        play_steps = bot.play(25, greedy=True)
        if len(play_steps) == 25:
            if np.mean(play_steps) > bot.env.target and greedy == False:
                greedy = True
                end = time.time()
                result['n_ep_greedy'] = ep
                result['n_step_greedy'] = steps
                result['mean_steps_greedy'] = np.mean(play_steps)
                result['time_greedy'] = end - start
            elif np.mean(play_steps) <= bot.env.target:
                end = time.time()
                result['n_ep_opti'] = ep
                result['n_step_opti'] = steps
                result['mean_steps_opti'] = np.mean(play_steps)
                result['time_opti'] = end - start
                break

    return result

algs = ['sarsa', 'naiveQ', 'watkinsQ']
algs1 = ['sarsa1', 'Q1']
tr = [0, 0.7, 0.8, 0.9, 1]
lr = [0.1, 0.2, 0.4, 0.6, 0.8]
eps = [0.1, 0.2]

bench = None
for a in algs:
    for t in tr:
        for l in lr:
            for e in eps:
                bot = Agent(env, algo=a, tr=t, lr=l, eps=e)
                r = pd.DataFrame(bench_algo(bot), index=[0])
                bench = pd.concat([bench, r])

for a in algs1:
    bot = Agent(env, algo=a)
    r = pd.DataFrame(bench_algo(bot), index=[0])
    bench = pd.concat([bench, r])
