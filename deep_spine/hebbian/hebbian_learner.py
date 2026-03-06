import numpy as np
from scipy import signal
import torch
import mujoco
import deep_spine.spine.filters.iir as iir
import deep_spine.utils.viz as viz

class HebbianLearner:
    def __init__(
        self,
        W,
        pre,
        post,
        path_to_log = None,
        gain = 1,
        learning_rate=0.01,
        lock_target = None,
        postive_only = False,
        min_weight = 0,
        max_weight = 1,
        Kmean = 1,
        Klearn = 1,
        synaptic_filter = {'order': 1, 'cutoff': 0.1, 'type': 'low'},
    ):
        self.W = W
        self.pre_synaptic = pre
        self.post_synaptic = post

        self.n_post, self.n_pre = W.shape

        self.min_weight = min_weight
        self.max_weight = max_weight
        self.gain = gain
        self.learning_rate = learning_rate
        self.learning_signal = None
        self.logfile = None
        self.data_cache = None
        self.lock_target = lock_target
        self.postive_only = postive_only
        if path_to_log:
            self.logfile = open(path_to_log, 'ab')

        self.learning_filter = iir.IirBatch(
            [0, Klearn], [1, Klearn-1], (self.n_post, self.n_pre)
        )
        self.mean_filter = iir.IirBatch(
            [0, Kmean], [1, Kmean-1], (self.n_post,)
        )
        b, a = signal.butter(
            synaptic_filter['order'],
            synaptic_filter['cutoff'],
            synaptic_filter['type']
        )
        self.synaptic_filter = iir.IirBatch(b, a, (self.n_post,))
        self.high_pass_post_synaptic = torch.zeros(self.n_post)
        self.mean_post_synaptic = torch.zeros(self.n_post)
        self.post_dev = torch.zeros(self.n_post)
        self.learning_signal = torch.zeros(self.n_post,self.n_pre)
        self.compensation = torch.zeros(self.n_post,self.n_pre)
        self.rate = torch.zeros(self.n_post,self.n_pre)
        self.dW = torch.zeros(self.n_post,self.n_pre)

    def set_logfile(self, path_to_log):
        if path_to_log:
            self.logfile = open(path_to_log, 'ab')
        else:
            self.logfile = None

    @torch.no_grad()
    def step(self):


        # --- FILTER STAGE (vectorized) ---
        post = np.asarray(self.post_synaptic, dtype=np.float64)
        pre = np.asarray(self.pre_synaptic, dtype=np.float64)

        high_pass = self.synaptic_filter.filter(post)       # (n_post,)
        mean_post = self.mean_filter.filter(post)           # (n_post,)

        self.high_pass_post_synaptic.copy_(torch.from_numpy(high_pass))
        self.mean_post_synaptic.copy_(torch.from_numpy(mean_post))

        post_dev = self.gain * high_pass - mean_post        # (n_post,)
        self.post_dev.copy_(torch.from_numpy(post_dev))

        # --- LEARNING SIGNAL STAGE (vectorized) ---
        W_np = self.W.numpy()                               # (n_post, n_pre)
        raw = np.outer(post_dev, pre) * W_np                # (n_post, n_pre)
        ls = self.learning_filter.filter(raw)               # (n_post, n_pre)
        self.learning_signal.copy_(torch.from_numpy(ls))

        # --- VECTORIZED WEIGHT UPDATE ---
        signal = self.learning_signal                    # (n, n)
        comp = torch.where(signal >= 0, 1 - self.W, self.W)
        rate = (self.mean_post_synaptic ** 4) * self.learning_rate  # (n,)
        rate = rate.view(-1, 1)                          # (n, 1) for broadcast

        dW = comp * signal * rate                        # (n, n)

        if self.lock_target:
            lock_mask = self.W >= self.lock_target
            dW = dW.masked_fill(lock_mask, 0)

        if self.postive_only:
            dW = dW.clamp(min=0)

        self.dW.copy_(dW)

        self.W.add_(self.dW)
        # self.W.clamp_(min=self.min_weight, max=self.max_weight)


# LOOP BASED CALC
#        pre = torch.tensor(self.pre_synaptic, dtype=torch.float32)
#        post = torch.tensor(self.post_synaptic, dtype=torch.float32)
#
#        # find weights larger than the 'lock_target'
#        if self.lock_target:
#            lock_bool = self.W > self.lock_target
#
#        # Hebbian learning
#        for i in range(self.n):
#            self.high_pass_post_synaptic[i] = torch.tensor(
#                self.synaptic_filter[i].filter(self.post_synaptic[i])
#            )
#            self.mean_post_synaptic[i] = torch.tensor(
#                self.mean_filter[i].filter(self.post_synaptic[i])
#            )
#            self.post_dev[i] = (
#                self.gain*self.high_pass_post_synaptic[i]
#                - self.mean_post_synaptic[i]
#            )
#            for j in range(self.n):
#                self.learning_signal[i, j] = (
#                    self.learning_filter[i][j].filter(
#                        self.pre_synaptic[j] * self.post_dev[i] * self.W[i, j]
#                    )
#                )
#                if self.learning_signal[i, j] >= 0:
#                    self.compensation[i, j] = 1 - self.W[i, j]
#                else:
#                    self.compensation[i,j] = self.W[i, j]
#
#                self.rate[i, j] = (
#                    self.mean_post_synaptic[i]**4 * self.learning_rate
#                )
#                self.dW[i, j] = (
#                    self.compensation[i, j]
#                    * self.learning_signal[i, j]
#                    * self.rate[i, j]
#                )
#                # if dW is negative but the current weight is larger than the
#                # lock_target, set dW to 0
#                if self.lock_target and lock_bool[i, j]:
#                    self.dW[i, j] = 0
#
#                if self.postive_only and self.dW[i, j] < 0:
#                    self.dW[i, j] = 0
#
#        self.W.add_(self.dW)
#        self.W.clamp_(min=self.min_weight, max=self.max_weight)
#
#        data = np.concatenate([
#            self.pre_synaptic,
#            self.post_synaptic,
#            self.high_pass_post_synaptic,
#            self.mean_post_synaptic,
#            self.learning_signal.flatten(),
#            self.compensation.flatten(),
#            self.rate.flatten(),
#            self.dW.flatten(),
#            self.W.flatten()
#        ])
#        self.data_cache = data
#        #data_str = ','.join(map('{:.6f}'.format, data))
#        #print(data_str, file=self.logfile)

        data = np.array([self.W.flatten()])
        if self.logfile:
            self.logfile.write(data.tobytes())
