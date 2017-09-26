import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt


class Rental:
    def __init__(self, max_car=20, max_move=5, r_rent=10, r_move=-2):
        """ Rental car env. Probs of return and rent are fixed"""
        self.max_car = max_car
        self.max_move = max_move
        self.r_rent = r_rent
        self.r_move = r_move
        self.mean_rent = [3, 4]
        self.mean_ret = [3, 2]
        self.reset()

    def reset(self):
        """
        the game begins with 5 cars in each location
        """
        self.observation = [5, 5]

    def valid_actions(self):
        low = max(-1 * self.max_car + self.observation[1],
                  -1 * self.observation[1], -1 * self.max_move)
        high = min(self.max_car - self.observation[0],
                   self.observation[0], self.max_move)

        return range(low, high + 1, 1)

    def is_action_valid(self, action):
        return action in self.valid_actions()

    def step(self, action):
        """
        action is an integer, positive for a move from a to b
        and negative otherwise
        """
        if not self.is_action_valid(action):
            raise Exception('This action is not authorized')

        reward = self.r_move * np.abs(action)

        self.observation[0] -= action
        self.observation[1] += action

        returned = [np.random.poisson(m) for m in self.mean_ret]
        rented = [np.random.poisson(m) for m in self.mean_rent]

        reward += min(rented[0], self.observation[0]) * self.r_rent
        reward += min(rented[1], self.observation[1]) * self.r_rent

        # you rent only car available
        self.observation[0] = max(self.observation[0] - rented[0], 0)
        self.observation[1] = max(self.observation[1] - rented[1], 0)

        # returned cars are available only the next day
        self.observation[0] = min(self.observation[0] + returned[0], 20)
        self.observation[1] = min(self.observation[1] + returned[1], 20)

        return self.observation, reward, False, {
               'returned':returned, 'rented': rented}


env = Rental()

values = np.zeros((env.max_car, env.max_car))

pois_a = poisson(env.mean_ret[0])
pois_b = poisson(env.mean_ret[1])

new_values = values.copy()

for i in range(env.max_car):
    for j in range(env.max_car):
        env.observation = [i, j]
        moves = env.valid_actions()
        value = 0
        for m in moves:
            i_ = i - m
            j_ = j + m
            value += env.r_move * np.abs(m) \
                    + env.r_rent * (
                        min(i_, env.mean_rent[0]) + min(j_, env.mean_rent[1]))

            for k in range(env.max_car - i):
                for l in range(env.max_car - j):
                    value += pois_a.pmf(k) * pois_b.pmf(l) * values[i+k, j+l]

        value /= len(moves)
        new_values[i,j] = value

values = new_values.copy()

