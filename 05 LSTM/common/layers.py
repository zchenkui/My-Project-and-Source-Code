import sys, os 
sys.path.append(os.pardir)
from common.np import *
from common.config import GPU 
from common.functions import softmax, cross_entropy_error


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, np.transpose(W))
        dW = np.dot(np.transpose(self.x), dout)
        self.grads[0][...] = dW
        return dx


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, _ = self.params
        dx = np.dot(dout, np.transpose(W))
        dW = np.dot(np.transpose(self.x), dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Softmax:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.y = None # output of softmax layer
        self.t = None # the labels (true answer)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:
            self.t = np.argmax(self.t, axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx /= batch_size
        return dx


class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self, x):
        out = 1. / (1. + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1. - self.out) * self.out
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = 1. / (1. + np.exp(-x))
        self.loss = cross_entropy_error(np.c_[1-self.y, self.y], self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y -self.t) * dout / batch_size
        return dx


class Embedding:
    def __init__(self, W): 
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None 

    def forward(self, idx): 
        W, = self.params
        self.idx = idx 
        out = W[idx]
        return out 

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0

        np.add.at(dW, self.idx, dout)