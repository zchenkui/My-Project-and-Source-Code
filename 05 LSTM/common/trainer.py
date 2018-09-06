import time 
import numpy
import matplotlib.pyplot as plt 
import sys, os 
sys.path.append(os.pardir) 
from common.util import clip_grads
from common.np import *

def remove_duplicate(params, grads):
    '''
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads

class Trainer: 
    def __init__(self, model, optimizer): 
        self.model = model 
        self.optimizer = optimizer
        self.loss_list = [] 
        self.eval_interval = None 
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20, verbose=True):
        data_size = len(x)
        max_iters = data_size // batch_size # max iterations used to run one epoch
        self.eval_interval = eval_interval
        model = self.model
        optimizer = self.optimizer
        total_loss = 0 
        loss_count = 0

        start_time = time.time() # record the start time 
        print("start training ...")
        for _ in range(max_epoch): # for each epoch
            # shuffle data
            idx = numpy.random.permutation(data_size)
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters): 
                x_batch = x[iters * batch_size : (iters + 1) * batch_size]
                t_batch = t[iters * batch_size : (iters + 1) * batch_size]

                loss = model.forward(x_batch, t_batch)
                model.backward() 
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss 
                loss_count += 1

                if (eval_interval is not None) and (iters % eval_interval == 0): 
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    if verbose:
                        print("| epoch %d |  iter %d / %d | time %d[s] | loss %.2f" 
                            % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss)) 
                    self.loss_list.append(float(avg_loss))
                    total_loss = 0 
                    loss_count = 0

            self.current_epoch += 1
        print("end training ...")
        print() 

    def plot(self, ylim=None):
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel("iterations (x" + str(self.eval_interval) + ")")
        plt.ylabel("loss")
        plt.show()


class RnnlmTrainer:
    def __init__(self, model, optimizer): 
        self.model = model 
        self.optimizer = optimizer
        self.time_idx = None 
        self.ppl_list = None 
        self.eval_interval = None 
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size): 
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i") 

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
        
        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35, max_grad=None, eval_interval=20, verbose=True):
        data_size = len(xs) 
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = [] 
        self.eval_interval = eval_interval
        model = self.model
        optimizer = self.optimizer 
        total_loss = 0
        loss_count = 0 

        start_time = time.time() 
        print("start training ...")
        for epoch in range(max_epoch): 
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                loss = model.forward(batch_x, batch_t)
                model.backward() 
                params, grads = remove_duplicate(model.params, model.grads) 
                if max_grad is not None: 
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss 
                loss_count += 1

                if (eval_interval is not None) and (iters % eval_interval) == 0: 
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time 
                    if verbose: 
                        print("| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f"
                         % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0
            
            self.current_epoch += 1 
        print("end training ...")
        print() 

    def plot(self, ylim=None): 
        x = numpy.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('perplexity')
        plt.show()