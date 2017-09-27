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
        low = max(-1 * (self.max_car - self.observation[0]),
                  -1 * self.observation[1], -1 * self.max_move)
        high = min(self.max_car - self.observation[1],
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


def state_value_calc(env, i, j, gamma, pois_rent, pois_ret, values):
    env.observation = [i, j]
    moves = env.valid_actions()
    value = 0
    for m in moves:
        i_ = i - m
        j_ = j + m
        reward = env.r_move * np.abs(m)
        value = 0

        # i_ +1 because of python's 0 based index
        for r0, r1 in np.ndindex((i_ + 1, j_ + 1)):
            i__ = i_ - r0
            j__ = j_ - r1
            prob = 1
            # if r0 < i_ : prob(rent = r0) = pmf(r0)
            # if r0 = i_ : prob(rent = r0) = cdf(r0 - 1)
            if r0 == i_:
                prob *= (1 - pois_rent[0].cdf(r0 - 1))
            else:
                prob *= pois_rent[0].pmf(r0)
            # if r1 < j_ : prob(rent = r1) = pmf(r1)
            # if r1 = j_ : prob(rent = r1) = cdf(r1 - 1)
            if r1 == j_:
                prob *= (1 - pois_rent[1].cdf(r1 - 1))
            else:
                prob *= pois_rent[1].pmf(r1)

            value += prob * (r0 + r1) * env.r_rent

            for k, l in np.ndindex((env.max_car - i__ + 1,
                                    env.max_car - j__ + 1)):
                prob = 1
                # same here. Returns get the nb of cars to env.max_car for
                # all values > env.max_car - i__. Hence cdf
                if k == env.max_car - i__:
                    prob *= (1 - pois_ret[0].cdf(k - 1))
                else:
                    prob *= pois_ret[0].pmf(k)
                if l == env.max_car - j__:
                    prob *= (1 - pois_ret[1].cdf(l - 1))
                else:
                    prob *= pois_ret[1].pmf(l)

                value += prob * values[i__+k, j__+l]

        value = reward + gamma * value

    value /= len(moves)
    return value

def states_value_func(env, values, delta_max, n_max=100):
    pois_ret = [poisson(env.mean_ret[a]) for a in range(2)]
    pois_rent =  [poisson(env.mean_rent[a]) for a in range(2)]

    new_values = values.copy()
    n = 0
    delta = 1

    while delta > delta_max and n < n_max:
        delta = 0
        for i, j in np.ndindex(np.shape(values)):
            new_values[i,j] = state_value_calc(env, i, j, 0.9,
                                               pois_rent, pois_ret, values)
            delta = max(delta, np.abs(new_values[i,j] - values[i,j]))

        values = new_values.copy()
        n += 1
        print("Iteration number: ", n)

    return values

env = Rental(10, 3)

values = np.zeros((env.max_car + 1, env.max_car + 1))
values = states_value_func(env, values, 0.1, 5)

## policy evaluation - ici le choix de la prochaine action est equiprobable

# make a function that return new_values and take env as arg

values = np.zeros((env.max_car + 1, env.max_car + 1))

pois_ret = [poisson(env.mean_ret[a]) for a in range(2)]
pois_rent =  [poisson(env.mean_rent[a]) for a in range(2)]

new_values = values.copy()
delta = 1

while delta > 0.1:
    delta = 0
    for i, j in np.ndindex(np.shape(values)):
        new_values[i,j] = state_value_calc(env, i, j)
        delta = max(delta, np.abs(new_values[i,j] - values[i,j]))

    values = new_values.copy()









while delta > 0.1:
    delta = 0
    # i want a 21 by 21 matrix. There can be 0 cars and 20 cars
    for i, j in np.ndindex(np.shape(values)):
        env.observation = [i, j]
        moves = env.valid_actions()
        value = 0
        for m in moves:
            i_ = i - m
            j_ = j + m
            value += env.r_move * np.abs(m)

            # i_ +1 because of python's 0 based index
            for r0, r1 in np.ndindex((i_ + 1, j_ + 1)):
                i__ = i_ - r0
                j__ = j_ - r1
                prob = 1
                # if r0 < i_ : prob(rent = r0) = pmf(r0)
                # if r0 = i_ : prob(rent = r0) = cdf(r0 - 1)
                if r0 == i_:
                    prob *= (1 - pois_rent[0].cdf(r0 - 1))
                else:
                    prob *= pois_rent[0].pmf(r0)
                # if r1 < j_ : prob(rent = r1) = pmf(r1)
                # if r1 = j_ : prob(rent = r1) = cdf(r1 - 1)
                if r1 == j_:
                    prob *= (1 - pois_rent[1].cdf(r1 - 1))
                else:
                    prob *= pois_rent[1].pmf(r1)

                value += prob * (r0 + r1) * env.r_rent

                for k, l in np.ndindex((env.max_car - i__ + 1,
                                        env.max_car - j__ + 1)):
                    prob = 1
                    # same here. Returns get the nb of cars to env.max_car for
                    # all values > env.max_car - i__. Hence cdf
                    if k == env.max_car - i__:
                        prob *= (1 - pois_ret[0].cdf(k - 1))
                    else:
                        prob *= pois_ret[0].pmf(k)
                    if l == env.max_car - j__:
                        prob *= (1 - pois_ret[1].cdf(l - 1))
                    else:
                        prob *= pois_ret[1].pmf(l)

                    value += prob * values[i__+k, j__+l]

        value /= len(moves)
        new_values[i,j] = value
        delta = max(delta, np.abs(new_values[i,j] - values[i,j]))

    values = new_values.copy()

plt.imshow(values, cmap='hot', interpolation='nearest')
plt.legend()
plt.show()

best_action = np.zeros((env.max_car, env.max_car))
new_best_action = best_action.copy()

for i in range(env.max_car):
    for j in range(env.max_car):
        env.observation = [i, j]
        moves = env.valid_actions()
        moves_value = []
        for m in moves:
            i_ = i - m
            j_ = j + m

            # calculate the mean over all possible next states for move m
            for k in range(env.max_car - i):
                for l in range(env.max_car - j):
                    value += pois_ret[0].pmf(k) * pois_ret[1].pmf(l) \
                           * values[i+k, j+l]

            moves_value.append(value)

        new_best_action[i, j] = moves[np.argmax(moves_value)]

best_action = new_best_action.copy()

plt.imshow(best_action, cmap='hot', interpolation='nearest')
plt.legend()
plt.show()
