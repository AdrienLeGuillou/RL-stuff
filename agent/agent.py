import numpy as np
# from gridworld.gridworld import Gridworld

exec(open("../gridworld/gridworld.py").read())

class Agent:
    def __init__(self, env, tr=0.9, lr=0.1, dr=0.9, eps=0.1):
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
        self._reset()

    def _reset(self):
        self.Q = np.zeros((len(self.S), len(self.A)))
        self.pi = np.zeros((len(self.S), len(self.A)))
        for s in range(len(self.S)):
            self._update_pi(s)

    def _update_pi(self, s):
        best = np.argmax(self.pi[s])
        self.pi[s] = 1
        self.pi[s] *= self.eps / len(self.A)
        self.pi[s][best] += 1 - self.eps

    def _pick_action(self, s, greedy=False):
        if greedy:
            return np.argmax(self.pi[s])
        else:
            return np.random.choice(range(len(self.A)), p=self.pi[s])

    def train(self, n=1):
        while n > 0:
            self.env.reset()
            Z = np.zeros((len(self.S), len(self.A)))
            s = self.env.get_state()

            # do stuff

            n -= 1

world = np.array([
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 9, 9, 1, 2, 2, 1, 0],
    [7, 0, 0, 9, 9, 1, 2, 8, 1, 0],
    [0, 0, 0, 9, 9, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
])


env = Gridworld(world, (1, 0))

bot = Agent(env)