import numpy as np
from scipy import signal
import torch
import mujoco
import deep_spine.spinal_layers.filters.iir as iir
import deep_spine.environments.viz as viz

class HebbianMonitorFigure:
    def __init__(
        self,
        xs,
        ys,
        H,
        W,
        T,
        sim_timestep,
        source
    ):
        N = int(T / sim_timestep)
        self.figs = [
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-H, W, H),
                ntraces=source.pre_synaptic.numel(),
                window=N,
                source=source.pre_synaptic,
                ylabel='pre',
                xlim=np.array([0, T]),
                ylim=np.array([0, 1])
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-2*H, W, H),
                ntraces=source.post_synaptic.numel(),
                window=N,
                source=source.post_synaptic,
                ylabel='post',
                xlim=np.array([0, T]),
                ylim=np.array([0, 1])
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-3*H, W, H),
                ntraces=source.post_dev.numel(),
                window=N,
                source=source.post_dev,
                ylabel='dev',
                xlim=np.array([0, T]),
                ylim=np.array([0, 1])
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-4*H, W, H),
                ntraces=source.learning_signal.numel(),
                window=N,
                source=source.learning_signal,
                ylabel='learn',
                xlim=np.array([0, T]),
                ylim=np.array([-0.001, 0.001]),
                yformat='%.5f',
                add_zero_line=True
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-5*H, W, H),
                ntraces=source.compensation.numel(),
                window=N,
                source=source.compensation,
                ylabel='comp',
                xlim=np.array([0, T]),
                ylim=np.array([-0.001, 0.001]),
                yformat='%.5f'
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-6*H, W, H),
                ntraces=source.rate.numel(),
                window=N,
                source=source.rate,
                ylabel='rate',
                xlim=np.array([0, T]),
                ylim=np.array([-0.0001, 0.0001]),
                yformat='%.5f'
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-7*H, W, H),
                ntraces=source.dW.numel(),
                window=N,
                source=source.dW,
                ylabel='dW',
                xlim=np.array([0, T]),
                ylim=np.array([-0.0001, 0.0001]),
                yformat='%.5f'
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-7*H - 20, W, 10),
                ntraces=0,
                window=N,
                xlim=np.array([-T, 0])
            )
        ]
        self.figs[-1].figure.yformat = ''
        self.figs[-1].figure.flg_ticklabel[0] = 1
        self.figs[-1].figure.flg_ticklabel[1] = 1

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

        self.pre_synaptic = torch.zeros(self.n)
        self.post_synaptic = torch.zeros(self.n)
        self.high_pass_post_synaptic = torch.zeros(self.n)
        self.mean_post_synaptic = torch.zeros(self.n)
        self.post_dev = torch.zeros(self.n)
        self.learning_signal = torch.zeros(self.n,self.n)
        self.compensation = torch.zeros(self.n,self.n)
        self.rate = torch.zeros(self.n,self.n)
        self.dW = torch.zeros(self.n,self.n)

    def set_logfile(self, path_to_logfile):
        if path_to_logfile:
            self.logfile = open(path_to_logfile, 'w')
        else:
            self.logfile = None

    def step(self, pre_synaptic, post_synaptic):
        self.pre_synaptic[:] = torch.tensor(pre_synaptic)
        self.post_synaptic[:] = torch.tensor(post_synaptic)

        # find weights larger than the 'lock_target'
        if self.lock_target:
            lock_bool = self.W > self.lock_target

        for i in range(self.n):
            self.high_pass_post_synaptic[i] = torch.tensor(
                self.hp_filter[i].filter(self.post_synaptic[i])
            )
            self.mean_post_synaptic[i] = torch.tensor(
                self.mean_filter[i].filter(self.post_synaptic[i])
            )
            self.post_dev[i] = (
                self.gain*self.high_pass_post_synaptic[i]
                - self.mean_post_synaptic[i]
            )
            for j in range(self.n):
                self.learning_signal[i, j] = (
                    self.learning_filter[i][j].filter(
                        self.pre_synaptic[j] * self.post_dev[i] * self.W[i, j]
                    )
                )
                if self.learning_signal[i, j] >= 0:
                    self.compensation[i, j] = 1 - self.W[i, j]
                else:
                    self.compensation[i,j] = self.W[i, j]

                self.rate[i, j] = (
                    self.mean_post_synaptic[i]**4 * self.learning_rate
                )
                self.dW[i, j] = (
                    self.compensation[i, j]
                    * self.learning_signal[i, j]
                    * self.rate[i, j]
                )
                # if dW is negative but the current weight is larger than the
                # lock_target, set dW to 0
                if self.lock_target and lock_bool[i, j]:
                    self.dW[i, j] = 0

                if self.postive_only and self.dW[i, j] < 0:
                    self.dW[i, j] = 0


#self.post_dev = self.gain * self.high_pass_post_synaptic - self.mean_post_synaptic
#
#learning_input = (
#    self.pre_synaptic[np.newaxis, :] * self.post_dev[:, np.newaxis] * self.W
#)
#self.learning_signal = np.vectorize(
#    lambda x, i, j: self.learning_filter[i][j].filter(x)
#)(
#    learning_input, *np.indices(self.learning_signal.shape)
#)
#
#self.compensation = np.where(
#    self.learning_signal >= 0, 
#    1 - self.W, 
#    self.W
#)
#
#self.rate = (self.mean_post_synaptic[:, np.newaxis] ** 4) * self.learning_rate
#self.dW = self.compensation * self.learning_signal * self.rate



        self.W.add_(self.dW)
        self.W.clamp_(min=self.min_weight, max=self.max_weight)

        data = np.concatenate([
            self.pre_synaptic,
            self.post_synaptic,
            self.high_pass_post_synaptic,
            self.mean_post_synaptic,
            self.learning_signal.flatten(),
            self.compensation.flatten(),
            self.rate.flatten(),
            self.dW.flatten(),
            self.W.flatten()
        ])
        self.data_cache = data
        data_str = ','.join(map('{:.6f}'.format, data))
        print(data_str, file=self.logfile)
        return self.dW
