import numpy as np

class Agent:
    def __init__(self, env, Qmod, algo="sarsa1", dr=0.9, eps=0.1, reset=True):
        """
        env = openAI gym env
        Qmod = Q(s, a) estimator
        lr = learning rate
        dr = discount rate
        eps = epsilon for the e-greedy policy
        """
        self.env = env
        self.dr = dr
        self.eps = eps
        self.A = np.arange(2)
        self.algo = algo
        self.Qmod = Qmod
        self._algo(algo)
        if reset:
            self._reset()


    def _reset(self):
        s = self.env.observation_space.sample()
        a = self.env.action_space.sample()
        self.Qmod.build_model(s, a)


    def _algo(self, algo):
        if algo == "sarsa1":
            self._ep_train = self._train_sarsa1


    def _pick_action(self, s, greedy=False):
        states = []
        actions = []

        for a in self.A:
            states.append(s)
            actions.append(a)

        Q = self.Qmod.predict(states, actions)

        if greedy:
            a = self.A[np.argmax(Q)]
        else:
            p = np.random.random()
            if p <= self.eps:
                a = np.random.choice(self.A)
            else:
                a = self.A[np.argmax(Q)]

        return a


    def _update_model(self, s, a, q_):
        self.Qmod.fit(s, a, q_)


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

            q_ = r + self.dr * self.Qmod.predict(s_, a_)
            self._update_model(s, a, q_)

            s, a = s_, a_
            steps += 1
            n -= 1

        return steps, done
