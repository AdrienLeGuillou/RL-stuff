import numpy as np

class Gambler:
    def __init__(self, prob_head):
        self.prob_head = prob_head
        self.reset()

    def reset(self):
        """observation is the capital at state s"""
        self.observation = 99

    def valid_actions(self):
        # never bet more than what will get you to 100$
        return range(1, min(self.observation, 100 - self.observation ) + 1)

    def is_action_valid(self, action):
        return action in self.valid_actions()

    def step(self, action):
        """action is the money bet"""
        reward = 0
        done = False

        p = np.random.random()
        if p <= self.prob_head:
           self.observation += action
        else:
            self.observation -= action

        if self.observation == 0:
            done = True
        elif self.observation == 100:
            done = False
            reward = 1


        return self.observation, reward, done, {
               'ph':self.prob_head, 'p': p}

env = Gambler(0.25)

values = np.zeros(101)
values[100] = 1
policy =  np.ones(101)

delta = 1
n = 0
while delta > 0.1 and n < 100:
    # don't use 0 and 100 because it's game over
    for s in range(1, 100):
        max_val = 0
        env.observation = s
        for a in env.valid_actions():
            val = env.prob_head * values[s + a] \
                + (1 - env.prob_head) * values[s - a]
            if val > max_val:
                max_val = val
                policy[s] = a

        delta = max(np.abs(max_val - values[s]), delta)
        values[s] = max_val
    n += 1

