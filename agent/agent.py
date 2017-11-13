import numpy as np
from gridworld.gridworld import Gridworld
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class Agent:
    def __init__(self, env, algo="sarsa", tr=0.9, lr=0.1, dr=0.9, eps=0.1):
        """
        tr = trace
        lr = learning rate
        dr = discount rate
        eps = epsilon for the e-greedy policy
        """
        self.env = env
        self.tr = tr
        self.lr = lr
        self.dr = dr
        self.eps = eps
        self.A = self.env.valid_actions()
        self.S = self.env.valid_states()
        self._algo(algo)
        self._reset()

    def _reset(self):
        self.Q = np.zeros((len(self.S), len(self.A)))

    def _algo(self, algo):
        if algo == "sarsa":
            self._ep_train = self._train_sarsa
        elif algo == "sarsa1":
            self._ep_train = self._train_sarsa1
        elif algo == "Q1":
            self._ep_train = self._train_Q1
        elif algo == "naiveQ":
            self._ep_train = self._train_naiveQ
        elif algo == "watkinsQ":
            self._ep_train = self._train_watkinsQ

    def _pick_action(self, s, greedy=False):
        if greedy:
            return np.argmax(self.Q[s])
        else:
            p = np.random.random()
            if p <= self.eps:
                return np.random.randint(len(self.A))
            else:
                return np.argmax(self.Q[s])

    def _s_from_state(self, s):
        return self.S.index(s)

    def _action_from_a(self, a):
        return self.A[a]

    def play(self, n=-1, greedy=False, display=False):
        if n < 0:
            n = self.env.target * 2
        l_steps = []
        while True:
            steps = 0

            s = self.env.reset()
            s = self._s_from_state(s)
            a = self._pick_action(s)
            done = False

            while not done and n > 0:
                s, r, done = self.env.step(self._action_from_a(a))
                s = self._s_from_state(s)
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

    def train(self, n=1):
        l_steps = []
        while n > 0:
            l_steps.append(self._ep_train())
            n -= 1

        return l_steps

    def _train_sarsa(self):
        Z = np.zeros((len(self.S), len(self.A)))
        done = False
        steps = 0

        s = self.env.reset()
        s = self._s_from_state(s)
        a = self._pick_action(s)

        while not done:
            s_, r, done = self.env.step(self._action_from_a(a))
            s_ = self._s_from_state(s_)
            a_ = self._pick_action(s_)
            delta = r + self.dr * self.Q[s_, a_] - self.Q[s, a]
            Z[s, a] = min(Z[s, a] + 1, 1)

            self.Q += self.lr * delta * Z
            Z *= self.dr * self.tr

            s, a = s_, a_

            steps += 1

        return steps

    def _train_sarsa1(self):
        done = False
        steps = 0

        s = self.env.reset()
        s = self._s_from_state(s)
        a = self._pick_action(s)

        while not done:
            s_, r, done = self.env.step(self._action_from_a(a))
            s_ = self._s_from_state(s_)
            a_ = self._pick_action(s_)
            delta = r + self.dr * self.Q[s_, a_] - self.Q[s, a]

            self.Q[s,a] += self.lr * delta
            s, a = s_, a_

            steps += 1

        return steps

    def _train_Q1(self):
        done = False
        steps = 0

        s = self.env.reset()
        s = self._s_from_state(s)
        a = self._pick_action(s)

        while not done:
            s_, r, done = self.env.step(self._action_from_a(a))
            s_ = self._s_from_state(s_)
            a_ = self._pick_action(s_)
            a_max = self._pick_action(s_, greedy=True)
            delta = r + self.dr * self.Q[s_, a_max] - self.Q[s, a]

            self.Q[s,a] += self.lr * delta
            s, a = s_, a_

            steps += 1

        return steps

    def _train_naiveQ(self):
        Z = np.zeros((len(self.S), len(self.A)))
        done = False
        steps = 0

        s = self.env.reset()
        s = self._s_from_state(s)
        a = self._pick_action(s)

        while not done:
            s_, r, done = self.env.step(self._action_from_a(a))
            s_ = self._s_from_state(s_)
            a_ = self._pick_action(s_)
            a_max = self._pick_action(s_, greedy=True)
            delta = r + self.dr * self.Q[s_, a_max] - self.Q[s, a]
            Z[s, a] = min(Z[s, a] + 1, 1)

            self.Q += self.lr * delta * Z
            Z *= self.dr * self.tr

            s, a = s_, a_

            steps += 1

        return steps

    def _train_watkinsQ(self):
        Z = np.zeros((len(self.S), len(self.A)))
        done = False
        steps = 0

        s = self.env.reset()
        s = self._s_from_state(s)
        a = self._pick_action(s)

        while not done:
            s_, r, done = self.env.step(self._action_from_a(a))
            s_ = self._s_from_state(s_)
            a_ = self._pick_action(s_)
            a_max = self._pick_action(s_, greedy=True)
            delta = r + self.dr * self.Q[s_, a_max] - self.Q[s, a]
            Z[s, a] = min(Z[s, a] + 1, 1)

            self.Q += self.lr * delta * Z
            if a_ == a_max:
                Z *= self.dr * self.tr
            else:
                Z *= 0

            s, a = s_, a_

            steps += 1

        return steps

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

env = Gridworld(world_hard, (-1, 0), 100)

bot = Agent(env, algo="sarsa")

ep_step = bot.train(200)


plt.plot(ep_step)
plt.plot(savgol_filter(ep_step, 55, 1, mode='nearest'))
plt.axhline(10 * env.target, c='red')
plt.axhline(env.target, c='green')
plt.yscale('log')
# plt.xscale('log')
plt.show()

bot.play(greedy=True, display=True)
