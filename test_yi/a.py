import numpy as np
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test():
    a = np.random.rand(75, 80, 44, 3)
    b = np.random.rand(75, 40, 22, 3)
    c = np.random.rand(75, 20, 11, 3)
    t0 = time.time()
    y = sigmoid(a)
    y = sigmoid(b)
    y = sigmoid(c)
    print(time.time() - t0)


test()
