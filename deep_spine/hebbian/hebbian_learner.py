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
        path_to_log = './hebbian-data',
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
        if path_to_log:
            self.logfile = open(path_to_log, 'ab')

        self.learning_filter = [
            [iir.IirFilt([0, Klearn], [1, Klearn-1]) for _ in range(self.n)]
            for _ in range(self.n)
        ]
        self.mean_filter = (
            [iir.IirFilt([0, Kmean], [1, Kmean-1]) for _ in range(self.n)]
        )
        b, a = signal.butter(
            synaptic_filter['order'],
            synaptic_filter['cutoff'],
            synaptic_filter['type']
        )
        self.synaptic_filter = (
            [iir.IirFilt(b, a) for _ in range(self.n)]
        )
        self.high_pass_post_synaptic = torch.zeros(self.n)
        self.mean_post_synaptic = torch.zeros(self.n)
        self.post_dev = torch.zeros(self.n)
        self.learning_signal = torch.zeros(self.n,self.n)
        self.compensation = torch.zeros(self.n,self.n)
        self.rate = torch.zeros(self.n,self.n)
        self.dW = torch.zeros(self.n,self.n)

    def set_logfile(self, path_to_log):
        if path_to_log:
            self.logfile = open(path_to_log, 'ab')
        else:
            self.logfile = None

    @torch.no_grad()
    def step(self):


        # --- FILTER STAGE (scalar loops) ---
        for i in range(self.n):
            post = self.post_synaptic[i]
            self.high_pass_post_synaptic[i] = self.synaptic_filter[i].filter(post)
            self.mean_post_synaptic[i] = self.mean_filter[i].filter(post)

        high_pass = self.high_pass_post_synaptic         # (n,)
        mean_post = self.mean_post_synaptic              # (n,)
        post_dev = self.gain * high_pass - mean_post     # (n,)
        self.post_dev.copy_(post_dev)

        # --- LEARNING SIGNAL STAGE (still needs loop due to per-filter state) ---
        for i in range(self.n):
            for j in range(self.n):
                self.learning_signal[i, j] = (
                    self.learning_filter[i][j].filter(
                        self.pre_synaptic[j] * post_dev[i] * self.W[i, j]
                    )
                )

        # --- VECTORIZED WEIGHT UPDATE ---
        signal = self.learning_signal                    # (n, n)
        comp = torch.where(signal >= 0, 1 - self.W, self.W)
        rate = (mean_post ** 4) * self.learning_rate     # (n,)
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
                viewport=mujoco.MjrRect(xs, int(ys-0.5*H), W, int(0.5*H)),
                ntraces=source.pre_synaptic.numel(),
                window=N,
                source=source.pre_synaptic,
                ylabel='pre',
                xlim=np.array([0, T]),
                ylim=np.array([0, 1])
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-H, W, int(0.5*H)),
                ntraces=source.post_dev.numel(),
                window=N,
                source=source.post_dev,
                ylabel='post*',
                xlim=np.array([0, T]),
                ylim=np.array([0, 1])
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-2*H, W, H),
                ntraces=source.learning_signal[0][:].numel(),
                window=N,
                source=source.learning_signal[0][:],
                ylabel='L_1j',
                xlim=np.array([0, T]),
                ylim=np.array([-0.001, 0.001]),
                yformat='%.5f',
                ysymmetric=True
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-3*H, W, H),
                ntraces=source.learning_signal[1][:].numel(),
                window=N,
                source=source.learning_signal[1][:],
                ylabel='L_2j',
                xlim=np.array([0, T]),
                ylim=np.array([-0.001, 0.001]),
                yformat='%.5f',
                ysymmetric=True
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-4*H, W, H),
                ntraces=source.learning_signal[2][:].numel(),
                window=N,
                source=source.learning_signal[2][:],
                ylabel='L_3j',
                xlim=np.array([0, T]),
                ylim=np.array([-0.001, 0.001]),
                yformat='%.5f',
                ysymmetric=True
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-5*H, W, H),
                ntraces=source.learning_signal[3][:].numel(),
                window=N,
                source=source.learning_signal[3][:],
                ylabel='L_4j',
                xlim=np.array([0, T]),
                ylim=np.array([-0.001, 0.001]),
                yformat='%.5f',
                ysymmetric=True
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-6*H, W, H),
                ntraces=source.learning_signal[4][:].numel(),
                window=N,
                source=source.learning_signal[4][:],
                ylabel='L_5j',
                xlim=np.array([0, T]),
                ylim=np.array([-0.001, 0.001]),
                yformat='%.5f',
                ysymmetric=True
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(xs, ys-7*H, W, H),
                ntraces=source.learning_signal[5][:].numel(),
                window=N,
                source=source.learning_signal[5][:],
                ylabel='L_6j',
                xlim=np.array([0, T]),
                ylim=np.array([-0.001, 0.001]),
                yformat='%.5f',
                ysymmetric=True
            ),
            viz.TracesFigure(
                viewport=mujoco.MjrRect(
                    xs+W, ys -int(1.85*H), int(W/3.5), int(1.85*H)
                ),
                ntraces=source.learning_signal[5][:].numel(),
                window=N,
                source=None,
                xlim=np.array([0, T]),
                ylim=np.array([-10, -1]),
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
        self.figs[-2].figure.flg_legend = 1

        self.figs[-2].figure.linename[0] = 'SF'
        self.figs[-2].figure.linename[1] = 'SE'
        self.figs[-2].figure.linename[2] = 'EF'
        self.figs[-2].figure.linename[3] = 'EE'
        self.figs[-2].figure.linename[4] = 'BF'
        self.figs[-2].figure.linename[5] = 'BE'
        self.figs[-2].figure.flg_ticklabel[0] = 0
        self.figs[-2].figure.flg_ticklabel[1] = 0
        self.figs[-2].figure.flg_extend = 0

        self.figs[1].figure.flg_extend = 0
