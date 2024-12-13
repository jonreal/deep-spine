from typing import Optional, Any
import gymnasium as gym
import numpy as np
import mujoco

from stable_baselines3.common.callbacks import BaseCallback
from deep_spine.environments.base_environment import BaseEnvironment

from deep_spine.spinal_layers.filters import iir, mutual_info

import deep_spine.utils.scope as scope

class Figures:
    def __init__(
        self,
        viewport: mujoco.MjrRect,
        ntraces: int,
        window: int,
        ylabel: str=None,
        xlim: np.ndarray = None,
        ylim: np.ndarray = None
    ):
        self.ylabel = ylabel
        self.ntraces = ntraces
        self.buffer = np.zeros((ntraces, window))
        self.window = window
        self.viewport = viewport
        self.figure = mujoco.MjvFigure()
        self.figure.figurergba = np.array([0, 0, 0, 0])
        self.figure.panergba = np.array([0, 0, 0, 0])
        self.figure.flg_ticklabel[0] = 0
        self.figure.linewidth = 1
        self.figure.minwidth = '0000000'
        self.yformat = '%.2f'
        for i in range(ntraces):
            self.figure.linepnt[i] = window
            self.figure.linedata[i][:2*window:2] = np.linspace(0, 1, window)
            self.figure.linedata[i][1:2*window:2] = self.buffer[i,:]
        if xlim is not None:
            self.figure.range[0][0] = xlim[0]
            self.figure.range[0][1] = xlim[1]
        if ylim is not None:
            self.figure.range[1][0] = ylim[0]
            self.figure.range[1][1] = ylim[1]

    def update(self, data):
        self.buffer[:,:-1] = self.buffer[:,1:]
        self.buffer[:,-1] = data
        for i in range(self.ntraces):
            self.figure.linedata[i][1:2*self.window:2] = self.buffer[i,:]


class Arm26Babble(BaseEnvironment, gym.Env):
    def __init__(
        self,
        path_to_logfile: Optional[str] = None,
        scope_port: int = None,
        task: Optional[str] = '',
        xcor_window: int = 10,
        muscle_strength_factor: float = 1,
        babble_duration_range_sec = np.array([0.01, 0.1]),
        babble_rise_time_range = np.array([0.5, 0.8]),
        babble_max_prob: float = 0.1,
        episode_length: int = 100,
        seed: Optional[int] = 12345,
        spinal_layer: Optional[Any] = None,
        supraspinal_decimation_factor: int = 1,
        xml_path: str = 'arm26.xml',
        frame_rate: Optional[int] = None,
        save_frames_to_path: Optional[str] = None,
        render_in_background: bool = False,
        render_lossless: bool = False,
        azimuth: int = 180,
        elevation: int = -90,
        distance: int = 2,
        lookat: np.ndarray = np.array([0.35, 0, 0.25]),
        ) -> None:

        self.type = 'arm26babble'
        self.task = task

        self.udpcast = None
        if scope_port:
            self.udpcast = scope.ScopeBroadcaster(port=scope_port)

        super().__init__(
            xml_path=xml_path,
            frame_rate=frame_rate,
            save_frames_to_path=save_frames_to_path,
            render_in_background=render_in_background,
            render_lossless=render_lossless,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            lookat=lookat
        )
        if path_to_logfile:
            self.logfile = (
                open(path_to_logfile,'w') if path_to_logfile else None
            )
        else:
            self.logfile = None

        self.spinal_layer = self.initialize_spinal_layer(spinal_layer)
        self.episode_length = episode_length
        self.simulation_steps_per_episode = (
            self.episode_length * self.simulation_sampling_rate
        )

        self.done = False
        self.eps = 1
        self.reward = 0
        self.eps_reward = 0
        self.seed = seed

        np.random.seed(self.seed)
        self.ic_rng = np.random.default_rng(self.seed)

        self.supraspinal_decimation_factor = supraspinal_decimation_factor
        self.max_muscle_strengths = self.model.actuator_gainprm[:,3].copy()
        self.set_muscle_strength_factor(muscle_strength_factor)

        self.babble_duration_range = babble_duration_range_sec
        self.babble_rise_time_range = babble_rise_time_range
        self.babble_max_prob = babble_max_prob

        self.babble_signal = np.zeros(self.spinal_layer.n,)
        self.babble_prob = np.zeros(self.spinal_layer.n,)
        self.babble_t0 = np.zeros(self.spinal_layer.n,)
        self.babble_amp = np.zeros(self.spinal_layer.n,)
        self.babble_duration = np.zeros(self.spinal_layer.n,)
        self.babble_rise_time = np.zeros(self.spinal_layer.n,)
        self.babble_filter = [
            iir.IirFilt(
                [0, self.babble_rise_time_range[0]],
                [1, self.babble_rise_time_range[0] - 1]
            ) for i in range(self.spinal_layer.n)
        ]

        # Cache for the tb log file
        self.log_eps = 0
        self.log_eps_reward = 0
        self.log_babble_step = np.zeros((self.spinal_layer.n, 2))
        self.log_babble_amp = np.zeros(self.spinal_layer.n,)

        # Figures
        self.figs = []
        T = 1
        H = 80;
        ys = 900
        self.N = int(T / self.model.opt.timestep)

        # qpos
        fig = Figures(
            viewport=mujoco.MjrRect(800, ys-H, 400, H),
            ntraces=2,
            window=self.N,
            ylabel='qpos',
            xlim=np.array([0, T]),
            ylim=np.array([-2, 2])
        )
        self.figs.append(fig)
        # qvel
        fig = Figures(
            viewport=mujoco.MjrRect(800, ys-2*H, 400, H),
            ntraces=2,
            window=self.N,
            ylabel='qvel',
            xlim=np.array([0, T]),
            ylim=np.array([-0.5, 0.5])
        )
        self.figs.append(fig)
        # length
        fig = Figures(
            viewport=mujoco.MjrRect(800, ys-3*H, 400, H),
            ntraces=6,
            window=self.N,
            ylabel='len',
            xlim=np.array([0, T]),
            ylim=np.array([-1, 1])
        )
        self.figs.append(fig)
        # velocity
        fig = Figures(
            viewport=mujoco.MjrRect(800, ys-4*H, 400, H),
            ntraces=6,
            window=self.N,
            ylabel='vel',
            xlim=np.array([0, T]),
            ylim=np.array([-0.25, 0.25])
        )
        self.figs.append(fig)
        # force
        fig = Figures(
            viewport=mujoco.MjrRect(800, ys-5*H, 400, H),
            ntraces=6,
            window=self.N,
            ylabel='force',
            xlim=np.array([0, T]),
            ylim=np.array([-20, 0])
        )
        self.figs.append(fig)
        # act_II
        fig = Figures(
            viewport=mujoco.MjrRect(800, ys-6*H, 400, H),
            ntraces=6,
            window=self.N,
            ylabel='act_II',
            xlim=np.array([0, T]),
            ylim=np.array([0, 1])
        )
        self.figs.append(fig)
        # act_Ia
        fig = Figures(
            viewport=mujoco.MjrRect(800, ys-7*H, 400, H),
            ntraces=6,
            window=self.N,
            ylabel='act_Ia',
            xlim=np.array([0, T]),
            ylim=np.array([0, 1])
        )
        self.figs.append(fig)
        # babble_signal
        fig = Figures(
            viewport=mujoco.MjrRect(800, ys-8*H, 400, H),
            ntraces=6,
            window=self.N,
            ylabel='babble',
            xlim=np.array([0, T]),
            ylim=np.array([0, 1])
        )
        self.figs.append(fig)
        # act_aMN
        fig = Figures(
            viewport=mujoco.MjrRect(800, ys-9*H, 400, H),
            ntraces=6,
            window=self.N,
            ylabel='MN',
            xlim=np.array([0, T]),
            ylim=np.array([0, 1])
            )
        self.figs.append(fig)
        # act_muscle
        fig = Figures(
            viewport=mujoco.MjrRect(800, ys-10*H, 400, H),
            ntraces=6,
            window=self.N,
            ylabel='act_MN',
            xlim=np.array([0, T]),
            ylim=np.array([0, 1])
        )
        self.figs.append(fig)
        # activation
        fig = Figures(
            viewport=mujoco.MjrRect(800, ys-11*H, 400, H),
            ntraces=6,
            window=self.N,
            ylabel='act',
            xlim=np.array([0, T]),
            ylim=np.array([0, 1])
        )
        self.figs.append(fig)

        #xlabel
        fig = Figures(
            viewport=mujoco.MjrRect(800, ys-11*H - 20, 400, 10),
            ntraces=0,
            window=self.N,
            xlim=np.array([-T, 0])
        )
        fig.figure.yformat = ''
        fig.figure.flg_ticklabel[0] = 1
        fig.figure.flg_ticklabel[1] = 1
        self.figs.append(fig)

    def set_muscle_strength_factor(self, factor):
        self.model.actuator_gainprm[:,3] = factor * self.max_muscle_strengths

    def initialize_spinal_layer(self, spinal_layer):
        spinal_layer.initialize(self.model)
        self.simulation_sampling_rate = 1 / self.model.opt.timestep
        mujoco.set_mjcb_control(spinal_layer.callback)
        return spinal_layer

    def _set_initial_conditions(self):
        self.done = False
        self.reward = 0
        self.eps_reward = 0
        #mj.mj_resetData(self.model, self.data)
        self.data.time = 0
        self.spinal_layer.set_descending_drive(np.zeros(self.spinal_layer.n,))
        self.babble_ic = self._gen_babble()
        #mj.mj_forward(self.model, self.data)

        large_prime = 78531
        self.seed = str(int(self.seed * large_prime * (self.eps + 1)))
        self.seed = int(self.seed[:5])
        self.ic_rng = np.random.default_rng(self.seed)
        self.simulation_steps = 0

    def _gen_babble(self):
        if self.task == 'babble':
            for i in range(self.spinal_layer.n):
                if (self.data.time
                    >= self.babble_duration[i] + self.babble_t0[i]
                ) or (
                    self.babble_prob[i] == 0
                ) or (
                    self.data.time == 0
                ):
                    self.babble_prob[i] = (
                        self.ic_rng.uniform(0, 1) <= self.babble_max_prob
                    )
                    if self.babble_prob[i]:
                        self.babble_t0[i] = self.data.time
                        self.babble_amp[i] = (
                            self.ic_rng.uniform(0, 1)
                        )
                        self.babble_duration[i] = self.ic_rng.uniform(
                            self.babble_duration_range[0],
                            self.babble_duration_range[1]
                        )
                        self.babble_rise_time[i] = self.ic_rng.uniform(
                            self.babble_rise_time_range[0],
                            self.babble_rise_time_range[1]
                        )
                        self.babble_filter[i] = iir.IirFilt(
                                [0, self.babble_rise_time[i]],
                                [1, self.babble_rise_time[i] - 1]
                        )
        else:
            raise ValueError(f"Invalid task: {self.task}")

    def _update_babble(self):
        self.babble_signal = np.array([
            self.babble_prob[i]
            * self.babble_amp[i]
            * self.babble_filter[i].filter(
                self.data.time <= self.babble_duration[i] + self.babble_t0[i]
            ) for i in range(self.spinal_layer.n)
        ])


    def _update_status(self):
        if self.simulation_steps >= self.simulation_steps_per_episode:
            self.done = True
            self.eps += 1

    def step(self, action):
        self.reward = 0
        self.spinal_layer.set_descending_drive(self.babble_signal)
        for steps in range(self.supraspinal_decimation_factor):
            mujoco.mj_step(self.model, self.data)
            self.simulation_steps += 1
            self._update_figures()
            self._render(self.figs)
            self._update_status()
            self._write_data_to_logfile()
            self._update_babble()
            self._gen_babble()

        self.eps_reward += (self.reward / self.simulation_sampling_rate)
        truncated = False
        info = {}
        return {}, self.reward, self.done, truncated, info

    def reset(self,seed=None,**kwargs):
        super().reset(seed=seed)
        #self._update_tb_log()
        self.done = False
        self.count = 0
        self._set_initial_conditions()
        info = {}
        return {}, {}

    def set_logfile(self, path_to_logfile):
        if path_to_logfile:
            self.logfile = open(path_to_logfile, 'w')
        else:
            self.logfile = None

    def _update_figures(self):

        # qpos
        self.figs[0].update(self.data.qpos)
        # qvel
        self.figs[1].update(self.data.qvel)
        # length
        self.figs[2].update(self.spinal_layer.length)
        # velocity
        self.figs[3].update(self.spinal_layer.velocity)
        # force
        self.figs[4].update(self.spinal_layer.force)
        # act_II
        self.figs[5].update(self.spinal_layer.act_II)
        # act_Ia
        self.figs[6].update(self.spinal_layer.act_Ia)
        # babble_signal
        self.figs[7].update(self.babble_signal)
        # act_aMN
        self.figs[8].update(self.spinal_layer.act_aMN)
        # act_muscle
        self.figs[9].update(self.spinal_layer.act_muscle)
        # activation
        self.figs[10].update(self.spinal_layer.activation)

    def _write_data_to_logfile(self):
        if self.logfile:
            data = np.concatenate([
                self.data.qpos,
                self.data.qvel,
                self.spinal_layer.length,
                self.spinal_layer.velocity,
                self.spinal_layer.force,
                self.spinal_layer.act_II,
                self.spinal_layer.act_Ia,
                self.babble_signal,
                self.spinal_layer.act_aMN,
                self.spinal_layer.act_muscle,
                self.spinal_layer.activation,
            ])
            if self.udpcast:
                self.udpcast.broadcast(data)
            data_str = ','.join(map('{:.6f}'.format, data))
            print(data_str, file=self.logfile)

class LogCallbackArm26Babble(BaseCallback):
    def __init__(self, env, window, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.window = window
        self.ep_rews = np.zeros(window)

    def _on_step(self) -> bool:
        # Identify new episode
        if self.env.data.time == 0.0 and self.env.eps > 0:
            self.ep_rews = np.concatenate(
              (self.ep_rews[1:], np.array([self.env.log_eps_reward]))
            )
            if self.env.eps >= self.window:
                self.logger.record(
                    f"eps/ep_rew_mean_{self.window}",
                    np.mean(self.ep_rews)
                )
            self.logger.record("eps/ep_rew", self.env.log_eps_reward)
            self.logger.record(f"eps/eps",self.env.log_eps)
            #for i in range(self.env.spinal_layer.n):
            #    self.logger.record(
            #        f"eps/b{i+1}_step1",
            #        self.env.log_babble_step[i][0]
            #    )
            #    self.logger.record(
            #        f"eps/b{i+1}_step2",
            #        self.env.log_babble_step[i][1]
            #    )
            #    self.logger.record(
            #        f"eps/b{i+1}_amp",
            #        self.env.log_babble_amp[i]
            #    )
            #self.logger.dump(self.num_timesteps)
        return True


