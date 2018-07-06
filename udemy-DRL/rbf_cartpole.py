import os
os.chdir('/home/jovyan/work/RL-stuff/udemy-DRL')

import numpy as np
import gym
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
#from sklearn.linear_model import SGDRegressor
import matplotlib
import matplotlib.pyplot as plt

class SGDRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 10e-2

    def partial_fit(self, X, Y):
        self.w += self.lr * (Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)


class Feature_transformer:
    def __init__(self, env):
        observation_sample = np.random.random((2000, 4)) * 2 - 1
        scaler = StandardScaler()
        features = FeatureUnion([
            ('rbf0', RBFSampler(gamma=5.0, n_components=1000)),
            ('rbf1', RBFSampler(gamma=5.0, n_components=1000)),
            ('rbf2', RBFSampler(gamma=5.0, n_components=1000)),
            ('rbf3', RBFSampler(gamma=5.0, n_components=1000))])

        transformer = Pipeline([('scaler', scaler),
                                ('features', features)])

        feature_example = transformer.fit(observation_sample)

        self.transformer = transformer

    def transform(self, observation):
        # make sure the observation is a 2D array
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        return self.transformer.transform(observation)


class Agent:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer
        self.make_models()

    def make_models(self):
        self.models = []
        x = self.feature_transformer.transform(env.reset())

        for i in range(env.action_space.n):
            model = SGDRegressor(x.shape[1])
            self.models.append(model)

    def predict(self, s):
        X = self.feature_transformer.transform(s)
        return np.array([m.predict(X)[0] for m in self.models])

    def update(self, s, a, G):
        X = self.feature_transformer.transform(s)
        self.models[a].partial_fit(X, [G])

    def pick_action(self, s, eps):
        if np.random.random() < eps:
            a = self.env.action_space.sample()
        else:
            a = np.argmax(self.predict(s))

        return a


def play_one(bot, eps, gamma):
    s = bot.env.reset()
    done = False
    total_reward = 0
    iters = 0

    while not done and iters < 2000:
        a = bot.pick_action(s, eps)
        s_, r, done, _ = bot.env.step(a)

        if done:
            r = -200

        G = r + gamma*np.max(bot.predict(s_))
        bot.update(s, a, G)
        s = s_

        if not done:
            total_reward += r
        iters += 1

    return total_reward

def plot_running_avg(total_rewards):
    n = len(total_rewards)
    running_avg = np.cumsum(total_rewards) / (np.array(range(n)) + 1)

    plt.plot(running_avg)
    plt.title('Running Average')
    plt.show()


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    ft = Feature_transformer(env)
    bot = Agent(env, ft)
    gamma = 0.99

    N = 300
    total_rewards = np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        total_rewards[n] = play_one(bot, eps, gamma)
        print(f'Episode {n+1}/{N}, total_reward = {total_rewards[n]}')


    plt.plot(total_rewards)
    plt.title("Rewards")

    plot_running_avg(total_rewards)

    env = gym.wrappers.Monitor(env, 'monitor/cartpole_rbf', force=True)
    bot.env = env
    play_one(bot, 0, gamma)

    env.close()
