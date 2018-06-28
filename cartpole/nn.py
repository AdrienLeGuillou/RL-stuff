import numpy as np

def reLU(x):
    return x * (x > 0)

def logistic(x):
    return 1 / (1 + np.exp(-x))

def feed_forward(x, W, B):
    """
    inputs:
        x: input (1, n)
        W: list of weights [(n, m), (m, l), ..., (l, 1)]
        B: list of biases [(1, m), (1, l), ..., (1, 1)]
    outputs:
        H[-1]: output layer
        H[:-1]: list of all layers [X, hidden1, hidden2, ...]
    """

    H = [x]
    for i in range(len(W)):
        H.append(H[i] @ W[i] + B[i])
        if True: #i != len(W) - 1:
            H[i + 1] = logistic(H[i + 1])

    return H[-1], H[:-1]


def train(x, t, W, B, lr=0.1):
    """
    inputs:
        x: input (1, n)
        t: target(1, 1)
        W: list of weights [(n, m), (m, l), ..., (l, 1)]
        B: list of biases [(1, m), (1, l), ..., (1, 1)]
    outputs:
        W: list of weights [(n, m), (m, l), ..., (l, 1)]
        B: list of biases [(1, m), (1, l), ..., (1, 1)]
    """

    y, H = feed_forward(x, W, B)
    delta = y - t # (0, 0)
    grad_Z = y # (1, 1)

    for i in np.arange(len(H)) + 1:
        W_i = W[-i].copy() # copy W[-i] to update grad_Z with the right Ws
                    # H[-i] : (1, k)
                    # H[-i].reshape((-1, 1)) : (k, 1)
        W[-i] -= H[-i].reshape((-1, 1)) @ grad_Z * delta * lr
        B[-i] -= grad_Z * delta * lr
        #grad_relu_H = H[-i] > 0 # (1, k)
        grad_relu_H = H[-i] * (1 - H[-i]) # logistic
        grad_y = grad_Z @ W[-i].T
        grad_Z = grad_relu_H * grad_y

    return W, B



def XOR(x):
    if (x[0] or x[1]) and not (x[0] and x[1]):
        return 1
    else:
        return 0

XOR([0, 0])

for i in range(10000):
    X = np.random.randint(2, size=(2))
    Y = XOR(X)
    W, B = train(X, Y, W, B, lr=0.1)
    y, H = feed_forward(X, W, B)
    #print(f"delta = {Y[i] - y}, W1 = {W[1].flatten()}")


ys = []
Ys = []
for i in range(100):
    X = np.random.randint(2, size=(2))
    Y = XOR(X)
    y, H = feed_forward(X, W, B)
    ys.append(y)
    Ys.append(Y)

ys = np.array(ys).flatten().reshape((1, -1))
Ys = np.array(Ys).flatten().reshape((1, -1))
np.mean((Ys - ys)**2)

np.concatenate([Ys.T, ys.T], axis=1)
