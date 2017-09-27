import numpy as np
from scipy.stats import poisson

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

def get_transition_rewards_probs(env):
    r_sas = np.zeros((env.max_car + 1, env.max_car + 1,     # matrix of s
                      env.max_move * 2 + 1,                 # possible actions
                      env.max_car + 1, env.max_car + 1))    # matrix of s'

    # probabilities of transitioning from s through a to s'
    p_sas = np.zeros((env.max_car + 1, env.max_car + 1,     # matrix of s
                      env.max_move * 2 + 1,                 # possible actions
                      env.max_car + 1, env.max_car + 1))    # matrix of s'

    pois_ret = [poisson(env.mean_ret[a]) for a in range(2)]
    pois_rent =  [poisson(env.mean_rent[a]) for a in range(2)]

    for i, j in np.ndindex((env.max_car + 1, env.max_car + 1)):
        env.observation = [i, j]
        moves = env.valid_actions()
        for m in moves:
            i_ = i - m
            j_ = j + m
            reward = env.r_move * np.abs(m)
            value = 0
            # removal of the rented cars
            # i_ +1 to have [0, i_] with (i_ included)
            for r0, r1 in np.ndindex((i_ + 1, j_ + 1)):
                i__ = i_ - r0
                j__ = j_ - r1
                prob_ = 1
                # if r0 < i_ : prob(rent = r0) = pmf(r0)
                # if r0 = i_ : prob(rent = r0) = cdf(r0 - 1)
                if r0 == i_:
                    prob_ *= (1 - pois_rent[0].cdf(r0 - 1))
                else:
                    prob_ *= pois_rent[0].pmf(r0)
                # if r1 < j_ : prob(rent = r1) = pmf(r1)
                # if r1 = j_ : prob(rent = r1) = cdf(r1 - 1)
                if r1 == j_:
                    prob_ *= (1 - pois_rent[1].cdf(r1 - 1))
                else:
                    prob_ *= pois_rent[1].pmf(r1)

                rents = (r0 + r1) * env.r_rent

                for k, l in np.ndindex((env.max_car - i__ + 1,
                                        env.max_car - j__ + 1)):
                    prob__ = 1
                    # same here. Returns get the nb of cars to env.max_car for
                    # all values > env.max_car - i__. Hence cdf
                    if k == env.max_car - i__:
                        prob__ *= (1 - pois_ret[0].cdf(k - 1))
                    else:
                        prob__ *= pois_ret[0].pmf(k)
                    if l == env.max_car - j__:
                        prob__ *= (1 - pois_ret[1].cdf(l - 1))
                    else:
                        prob__ *= pois_ret[1].pmf(l)

                    # add env.max_car to m so m in [0, max_car*2]
                    a =  m + env.max_move
                    r_sas[i, j, a, i__ + k, j__ + l] = reward + rents
                    p_sas[i, j, a, i__ + k, j__ + l] = prob__ * prob_

    return r_sas, p_sas


def evaluate_policy(env, policy, state_value, r_sas, p_sas,
                    delta_max=1, n_max=3, gamma=0.9):
    delta = delta_max + 1
    n = 0
    while delta > delta_max and n < n_max:
        for i, j in np.ndindex(np.shape(state_value)):
            temp = state_value[i, j]
            value = 0
            for k, l in np.ndindex(np.shape(state_value)):
                a = policy[i, j] + env.max_car
                value += p_sas[i, j, a, k, l] \
                       * (r_sas[i, j, a, k, l] + gamma * state_value[k, l])

        state_value[i, j] = value
        delta = max(delta, np.abs(temp, state_value[i, j]))

def improve_policy(env, policy, state_value, r_sas, p_sas, gamma=0.9):
    stable = True
    for i, j in np.ndindex(np.shape(state_value)):
        temp = policy[i, j]
        moves = env.valid_actions()
        action_return = []
        for m in moves:
            value = 0
            for k, l in np.ndindex(np.shape(state_value)):
                a = m + env.max_move
                value += p_sas[i, j, a, k, l] \
                       * (r_sas[i, j, a, k, l] + gamma * state_value[k, l])

            action_return.append(value)

        policy[i, j] = moves[np.argmax(action_return)]

        if policy[i, j] != temp:
            stable = False

    return stable

def initialize(env):
    policy = np.zeros((env.max_car + 1, env.max_car + 1))
    state_value = np.zeros((env.max_car + 1, env.max_car + 1))

    return policy, state_value

#create the env
env = Rental()

# estimated reward for transitioning from s through a to s'
print("Computing the states action states transition \
       rewards and probabilities")
r_sas, p_sas = get_transition_rewards_probs(env)

print("Initializing the policy and the states values")
policy, state_value = initialize(env)

n = 0
while True:
    print("Evaluting policy: ", n)
    n += 1
    evaluate_policy(env, policy, state_value, r_sas, p_sas)
    # check if improve policy returns True (stable policy) then stop
    print("Improving the policy")
    if improve_policy(env, policy, state_value, r_sas, p_sas) and n < 10:
        if n >-= 10:
            print("Too many iterations. Stopping")
        else:
            print("The policy is now stable")
        break