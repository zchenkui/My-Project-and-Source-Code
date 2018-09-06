import numpy as np 

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    if np.ndim(x) == 2:
        x = x - np.amax(x, axis=1, keepdims=True)
        x = np.exp(x)
        x /= np.sum(x, axis=1, keepdims=True)
    elif np.ndim(x) == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x

def cross_entropy_error(y, t):
    if np.ndim(y) == 1:
        t = np.reshape(t, newshape=(1, t.size))
        y = np.reshape(y, newshape=(1, y.size))

    if t.size == y.size:
        t = np.argmax(t, axis=1)

    batch_size = y.shape[0]

    return (-1 * np.sum(np.log(y[np.arange(batch_size), t] + 1e-7))/batch_size)