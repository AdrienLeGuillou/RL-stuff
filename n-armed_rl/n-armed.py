import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, m, stationnary=True):
        self.m = m
        self.n = 0
        self.stationnary = stationnary

    def pull(self):
        self.n += 1

        if not self.stationnary:
            pass

        return np.random.randn() + self.m


class N_armed:
    def __init__(self, bandits, random_walk=False):
        self.bandits = bandits
        self.random_walk = random_walk

    def change_m(self):
        for b in self.bandits:
            b.m += np.random.randn()

    def play(self, k):
        if self.random_walk:
            self.change_m()
        return self.bandits[k].pull()


def agent(env, steps, **kwargs):
    policy = kwargs['policy']
    n_bandits = len(env.bandits)
    if 'alpha' in kwargs:
        rolling_mean = kwargs['rolling_mean']
    else:
        rolling_mean = False

    rewards = np.zeros(steps)
    means = np.zeros(n_bandits)
    Ns = np.zeros(n_bandits)
    probs = np.zeros(n_bandits)

    if policy in [
        'eps_greedy', 'greedy', 'optimistic_greedy', 'eps_greedy_decay']:

        eps = 0
        decay = False

        if policy in ['eps_greedy', 'eps_greedy_decay']:
            eps = kwargs['eps']

        if policy == 'eps_greedy_decay':
            decay = True

        if policy == 'optimistic_greedy':
            means = np.ones(n_bandits) * kwargs['init']

        def update_probs():
            if decay:
                step_eps = eps / max(np.sum(Ns) / 2, 1)
            else:
                step_eps = eps

            best = np.argmax(means)
            probs = np.zeros(n_bandits) + step_eps / n_bandits
            probs[best] += 1 - step_eps

            return probs

    elif policy == 'softmax':
        tau = kwargs['tau']

        def update_probs():
            probs = [np.exp(mean / tau) for mean in means]
            probs = probs / np.sum(probs)

            return probs

    probs = update_probs()

    for i in range(steps):

        j = np.random.choice(n_bandits, p=probs)

        rewards[i] = env.play(j)
        Ns[j] += 1
        if not rolling_mean:
            alpha = (1 / Ns[j])
        else:
            alpha = rolling_mean

        means[j] = means[j] + alpha * (rewards[i] - means[j])
        probs = update_probs()

    return np.cumsum(rewards) / np.arange(steps)


m_list = [0, 0, 0, 0, 0]
n = 10000

np.random.seed(0)
env = N_armed([Bandit(m) for m in m_list], True)
exp1 = agent(env, n, policy='eps_greedy', eps=0.1)

np.random.seed(0)
env = N_armed([Bandit(m) for m in m_list], True)
exp2 = agent(env, n, policy='eps_greedy', eps=0.01)

np.random.seed(0)
env = N_armed([Bandit(m) for m in m_list], True)
exp3 = agent(env, n, policy='greedy')

np.random.seed(0)
env = N_armed([Bandit(m) for m in m_list], True)
exp4 = agent(env, n, policy='softmax', tau=1.5)

np.random.seed(0)
env = N_armed([Bandit(m) for m in m_list], True)
exp5 = agent(env, n, policy='optimistic_greedy', init=10)

np.random.seed(0)
env = N_armed([Bandit(m) for m in m_list], True)
exp6 = agent(env, n, policy='eps_greedy_decay', eps=1)

np.random.seed(0)
env = N_armed([Bandit(m) for m in m_list], True)
exp7 = agent(env, n, policy='eps_greedy', eps=0.1, rolling_mean=1/30)

plt.plot(exp1, label='eps = 0.1')
plt.plot(exp2, label='eps = 0.01')
plt.plot(exp3, label='greedy')
plt.plot(exp4, label='softmax')
plt.plot(exp5, label='optimistic 10')
plt.plot(exp6, label='eps with decay')
plt.plot(exp7, label='eps = 0.1 | rolling means')
for b in env.bandits:
    plt.axhline(b.m)
plt.xscale('log')
plt.legend()
plt.show()
