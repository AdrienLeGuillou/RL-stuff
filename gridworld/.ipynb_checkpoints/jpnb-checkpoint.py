from gridworld import Gridworld
import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, env):
        self.env = env
        self.position = self.env.position
        self.nb_a = len(env.valid_actions())
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 1
        self.pi = {}
        self.a = self._pick_action(self.position)
        self.Q = {}

    def _pick_action(self, state, greedy=False):
        if not state in self.pi:
            self.pi[state] = np.ones(self.nb_a)
            self.pi[state] *= self.epsilon / self.nb_a
            i = np.random.choice(self.nb_a)
            self.pi[state][i] += 1 - self.epsilon

        if greedy:
            return np.argmax(self.pi[state])
        else:
            return np.random.choice(range(self.nb_a), p=self.pi[state])

    def _update_Q(self, s, a, r, s_):
        sa = (s, a)
        a_ = self._pick_action(s_)
        sa_ = (s_, a_)

        if not sa in self.Q:
            self.Q[sa] = 0
        if not sa_ in self.Q:
            self.Q[sa_] = 0

        self.Q[sa] += self.alpha * (r + self.gamma * self.Q[sa_])

        return a_

    def _update_pi(self, s):
        sas = []
        for a in range(self.nb_a):
            sa = (s, a)
            if not sa in self.Q:
                self.Q[sa] = 0
            sas.append(self.Q[sa])
        g = np.argmax(sas)

        self.pi[s] = np.ones(self.nb_a)
        self.pi[s] *= self.epsilon / self.nb_a
        self.pi[s][g] += 1 - self.epsilon

    def train(self, n):
        counter = 0
        ep = 0
        while counter < n:
            done = False
            self.env.reset()
            while not done:
                s = self.position
                a = self.a
                s_, r, done = self.env.step(self.env.valid_actions()[a])

                self.position = s_
                self.a = self._update_Q(s, a, r, s_)
                self._update_pi(s)
                counter += 1
            print("episode", ep, "done")
                
        print('training done')


    def play(self, n, display=False):
        for _ in range(n):
            self.env.reset()
            done = False
            pos = []
            while not done:
                a = self._pick_action(self.position)
                self.position, r, done = self.env.step(
                                            self.env.valid_actions()[a])
                pos.append(self.position)

            print(len(pos))
            if display:
                state = self.env.world.copy()
                for p in pos:
                    state[p[0], p[1]] = 4
                state[self.env.end[0], self.env.end[1]] = 8

                plt.imshow(state, cmap='hot')
                plt.show()


world = np.array([
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 9, 9, 9, 2, 2, 1, 0],
    [7, 0, 0, 9, 9, 9, 2, 8, 1, 0],
    [0, 0, 0, 9, 9, 9, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
])

world_simple = np.array([
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [7, 0, 0, 1, 1, 1, 2, 8, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
])


env = Gridworld(world_simple, (1, 0))
bot = Agent(env)

bot.train(100)

bot.play(1)
