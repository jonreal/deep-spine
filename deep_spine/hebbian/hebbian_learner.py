import numpy as np
from scipy import signal
import torch
import deep_spine.spinal_layers.filters.iir as iir

class HebbianLearner:
    def __init__(
        self,
        W,
        path_to_logfile = './hebbian-data',
        gain = 1,
        learning_rate=0.01,
        lock_target = None,
        postive_only = False,
        min_weight = 0,
        max_weight = 1,
        Kmean = 1,
        Klearn = 1,
        hp_cut = 1
    ):
        self.W = W
        self.n = W.shape[0]
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.gain = gain
        self.learning_rate = learning_rate
        self.learning_signal = None
        self.logfile = None
        self.data_cache = None
        self.lock_target = lock_target
        self.postive_only = postive_only
        if path_to_logfile:
            self.logfile = open(path_to_logfile, 'w')

        self.learning_filter = [
            [iir.IirFilt([0, Klearn], [1, Klearn-1]) for _ in range(self.n)]
            for _ in range(self.n)
        ]
        self.mean_filter = (
            [iir.IirFilt([0, Kmean], [1, Kmean-1]) for _ in range(self.n)]
        )
        b, a = signal.butter(1, hp_cut, 'high')
        self.hp_filter = (
            [iir.IirFilt(b, a) for _ in range(self.n)]
        )

    def set_logfile(self, path_to_logfile):
        if path_to_logfile:
            self.logfile = open(path_to_logfile, 'w')
        else:
            self.logfile = None

    def step(self, pre_synaptic, post_synaptic):
        high_pass_post_synaptic = torch.tensor([
            self.hp_filter[i].filter(post_synaptic[i])
            for i in range(self.n)
        ])
        mean_post_synaptic = torch.tensor([
            self.mean_filter[i].filter(post_synaptic[i])
            for i in range(self.n)
        ])
        learning_signal = torch.zeros((self.n, self.n))
        compensation = torch.zeros((self.n, self.n))
        rate = torch.zeros((self.n, self.n))
        dW = torch.zeros((self.n, self.n))

        # find weights larger than the 'lock_target'
        if self.lock_target:
            lock_bool = self.W > self.lock_target

        for i in range(self.n):
            for j in range(self.n):
                learning_signal[i, j] = (
                    self.learning_filter[i][j].filter(
                        pre_synaptic[j]
                        * (self.gain*high_pass_post_synaptic[i]
                           - mean_post_synaptic[i]
                        )
                        * self.W[i, j]
                    )
                )
                if learning_signal[i, j] >= 0:
                    compensation[i, j] = 1 - self.W[i, j]
                else:
                    compensation[i,j] = self.W[i, j]

                rate[i, j] = mean_post_synaptic[i]**4 * self.learning_rate
                dW[i, j] = (
                    compensation[i, j] * learning_signal[i, j] * rate[i, j]
                )

                # if dW is negative but the current weight is larger than the
                # lock_target, set dW to 0
                if self.lock_target and lock_bool[i, j]:
                    dW[i, j] = 0

                if self.postive_only and dW[i, j] < 0:
                    dW[i, j] = 0

        self.W.add_(dW)
        torch.clamp_(self.W, self.min_weight, self.max_weight)

        data = np.concatenate([
            pre_synaptic,
            post_synaptic,
            high_pass_post_synaptic,
            mean_post_synaptic,
            learning_signal.flatten(),
            compensation.flatten(),
            rate.flatten(),
            dW.flatten(),
            self.W.flatten()
        ])
        self.data_cache = data
        data_str = ','.join(map('{:.6f}'.format, data))
        print(data_str, file=self.logfile)
        return dW
