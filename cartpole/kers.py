import numpy as np
#import gym
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

mod_q = Sequential()
mod_q.add(Dense(20, activation='sigmoid', input_dim=2))
mod_q.add(Dense(30, activation='sigmoid'))
mod_q.add(Dense(10, activation='sigmoid'))
mod_q.add(Dense(1, activation='sigmoid'))

mod_q.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])

def XOR(x):
    if (x[0] or x[1]) and not (x[0] and x[1]):
        return 1
    else:
        return 0

X = np.random.randint(2, size=(2))
Y = XOR(X)

mod_q.predict(X.reshape(1,2))
mod_q.fit(X.reshape(1, 2), np.array([Y])
for i in range(10000):
    X = np.random.randint(2, size=(2))
    Y = XOR(X)
    mod_q.fit(X.reshape(1, 2), np.array([Y]), verbose=False)
