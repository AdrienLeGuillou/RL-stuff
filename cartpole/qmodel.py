import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

class Qmodel:
    def __init__(self):
        pass


    def build_model(self, s, a):
        x = self._sa2x(s, a)

        self.mod = Sequential()
        self.mod.add(Dense(30, activation='sigmoid',
                           input_dim=x.shape[1],
                           kernel_initializer='zeros',
                           bias_initializer='zeros'))
        self.mod.add(Dense(30, activation='sigmoid'))
        self.mod.add(Dense(1, activation='linear'))

        self.mod.compile(loss='mse',
                         optimizer='adam'])


    def fit(self, s, a, q_):
        x = self._sa2x(s, a)
        y = self._q2y(q_)
        self._fit(x, y)


    def predict(self, s, a):
        x = self._sa2x(s, a)
        y = self._predict(x)

        return self._y2q(y)


    def _sa2x(self, s, a):
        if not isinstance(s, list):
            x = np.concatenate((s, np.array([a])))
            x = x.reshape(1, -1)
        else:
            s = np.array(s)
            a = np.array(a).reshape(-1, 1)
            x = np.concatenate((s, a), axis=1)

        return x


    def _y2q(self, y):
        return y


    def _q2y(self, q):
        return q


    def _fit(self, x, y):
        return self.mod.fit(x, y, verbose=False)


    def _predict(self, x):
        return self.mod.predict(x)
