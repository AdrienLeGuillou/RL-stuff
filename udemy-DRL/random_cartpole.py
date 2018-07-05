import os
os.chdir('/home/jovyan/work/RL-stuff/udemy-DRL')

import numpy as np
import gym

def get_action(s, w):
    return s.dot(w) > 0

def play_episode(env, w):
    s = env.reset()
    done = False
    steps = 0

    while(not done):
        a = get_action(s, w)
        s, r, done, _ = env.step(a)
        steps += 1

    return steps

def random_params():
    return np.random.random(4) * 2 - 1

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    w_star = random_params()
    max_steps = 0

    for _ in range(100):
        w = random_params()
        steps = play_episode(env, w)
        if steps > max_steps:
            max_steps = steps
            w_star = w

    print(max_steps)

    env = gym.wrappers.Monitor(env, 'monitor/cartpole_random', force=True)
    steps = 0
    for _ in range(100):
        steps += play_episode(env, w_star)

    env.close()

    print(f'avg steps for w* = {steps / 100}')
