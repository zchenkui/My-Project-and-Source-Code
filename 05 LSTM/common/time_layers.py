import sys, os 
sys.path.append(os.pardir) 
from common.np import *
from common.layers import *
from common.functions import *


class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None 

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params 
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b 
        h_next = np.tanh(t)
        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params 
        x, h_prev, h_next = self.cache

        dt = dh_next * (1.0 - h_next**2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(np.transpose(h_prev), dt)
        dh_prev = np.dot(dt, np.transpose(Wh)) 
        dWx = np.dot(np.transpose(x), dt)
        dx = np.dot(dt, np.transpose(Wx))   

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db 

        return dx, dh_prev 


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)] 
        self.layers = None 
        self.h = None 
        self.dh = None 
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = np.shape(xs)
        D, H = np.shape(Wx) 

        self.layers = [] 
        hs = np.empty(shape=(N, T, H)) 

        if not self.stateful or self.h is None: 
            self.h = np.zeros((N, H)) 

        for t in range(T): 
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h) 
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs 

    def backward(self, dhs): 
        Wx, Wh, b = self.params
        N, T, H = np.shape(dhs) 
        D, H = np.shape(Wx) 

        dxs = np.empty((N, T, D))
        dh = 0
        grads = [0, 0, 0] 

        for t in reversed(range(T)): 
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh) 
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads): 
                grads[i] += grad

        for i, grad in enumerate(grads): 
            self.grads[i][...] = grad 
        self.dh = dh 

        return dxs 

    def set_state(self, h): 
        self.h = h 
    
    def reset_state(self):
        self.h = None 


class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None 
        self.W = W 

    def forward(self, xs): 
        N, T = np.shape(xs)
        _, D = np.shape(self.W) 

        out = np.empty((N, T, D)) 
        self.layers = [] 

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t]) 
            self.layers.append(layer) 

        return out 

    def backward(self, dout):
        T = np.shape(dout)[1]
        grad = 0

        for t in range(T): 
            layer = self.layers[t]
            layer.backward(dout[:, t, :]) 
            grad += layer.grads[0]

        self.grads[0][...] = grad


class TimeAffine:
    def __init__(self, W, b): 
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None 

    def forward(self, x):
        N, T, D = np.shape(x)
        W, b = self.params

        rx = np.reshape(x, newshape=(N * T, -1)) 
        out = np.dot(rx, W) + b 
        out = np.reshape(out, newshape=(N, T, -1))
        self.x = x 
        return out 

    def backward(self, dout): 
        x = self.x 
        N, T, D = np.shape(x) 
        W, b = self.params

        dout = np.reshape(dout, newshape=(N * T, -1))
        rx = np.reshape(x, newshape=(N * T, -1))

        db = np.sum(dout, axis=0)
        dW = np.dot(np.transpose(rx), dout)
        dx = np.dot(dout, np.transpose(W))
        dx = np.reshape(dx, newshape=(N, T, D))

        self.grads[0][...] = dW 
        self.grads[1][...] = db

        return dx 


class TimeSoftmaxWithLoss:
    def __init__(self): 
        self.params, self.grads = [], [] 
        self.cache = None 
        self.ignore_label = -1 

    def forward(self, xs, ts): 
        N, T, V = np.shape(xs) 

        if np.ndim(ts) == 3:
            ts = np.argmax(ts, axis=2)

        mask = (ts != self.ignore_label)

        xs = np.reshape(xs, newshape=(N * T, V))
        ts = np.reshape(ts, newshape=(N * T)) 
        mask = np.reshape(mask, newshape=(N * T)) 

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask 
        loss = -1 * np.sum(ls)
        loss  /= np.sum(mask)

        self.cache = (ts, ys, mask, (N, T, V))

        return loss 

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys 
        dx[np.arange(N * T), ts] -= 1
        dx *= dout 
        dx /= np.sum(mask) 
        dx *= mask[:, np.newaxis]
        dx = np.reshape(dx, newshape=(N, T, V)) 

        return dx 


class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None 

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params 
        N, H = np.shape(h_prev) 

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b 
        f = A[:, 0 : H]
        g = A[:, H : 2 * H]
        i = A[:, 2 * H : 3 * H] 
        o = A[:, 3 * H : ]

        f = sigmoid(f)
        g = np.tanh(g) 
        i = sigmoid(i) 
        o = sigmoid(o)

        c_next = c_prev * f + g * i 
        h_next = np.tanh(c_next) * o 

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next): 
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        ds = dc_next + dh_next * o * (1 - np.tanh(c_next)**2)
        dc_prev = ds * f

        do = dh_next * np.tanh(c_next) 
        di = ds * g 
        dg = ds * i 
        df = c_prev * ds

        do *= o * (1 - o)
        di *= i * (1 - i) 
        dg *= (1 - g**2) 
        df *= f * (1 - f) 

        dA = np.hstack((df, dg, di, do)) 

        db = np.sum(dA, axis=0) 
        dWh = np.dot(np.transpose(h_prev), dA)
        dh_prev = np.dot(dA, np.transpose(Wh))
        dWx = np.dot(np.transpose(x), dA) 
        dx = np.dot(dA, np.transpose(Wx)) 

        self.grads[0][...] = dWx 
        self.grads[1][...] = dWh
        self.grads[2][...] = db 

        return dx, dh_prev, dc_prev  


class TimeLSTM: 
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)] 
        self.layers = None 

        self.h, self.c = None, None 
        self.dh = None 
        self.stateful = stateful

    def forward(self, xs): 
        Wx, Wh, b = self.params 
        N, T, D = np.shape(xs) 
        H = np.shape(Wh)[0] 

        self.layers = []
        hs = np.empty(shape=(N, T, H), dtype="f")

        if (not self.stateful) or (self.h is None): 
            self.h = np.zeros(shape=(N, H), dtype="f")
        if (not self.stateful) or (self.c is None): 
            self.c = np.zeros(shape=(N, H), dtype="f")

        for t in range(T): 
            layer = LSTM(Wx, Wh, b) 
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c) 
            hs[:, t, :] = self.h 

            self.layers.append(layer) 
        
        return hs 

    def backward(self, dhs): 
        Wx, Wh, b = self.params
        N, T, H = np.shape(dhs) 
        D = np.shape(Wx)[0]

        dxs = np.empty(shape=(N, T, D), dtype="f")
        dh, dc = 0, 0 

        grads = [0, 0, 0]
        for t in reversed(range(T)): 
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dh + dhs[:, t, :], dc) 
            dxs[:, t, :] = dx 
            for i, grad in enumerate(layer.grads): 
                grads[i] += grad

        for i, grad in enumerate(grads): 
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c 

    def reset_state(self): 
        self.h, self.c = None, None  