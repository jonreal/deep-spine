from typing import Optional, Any
import gymnasium as gym
import numpy as np
import mujoco
from stable_baselines3.common.callbacks import BaseCallback
from deep_spine.environments.base_environment import BaseEnvironment
import deep_spine.environments.viz as viz
from deep_spine.spinal_layers.filters import iir
import gzip

class Arm26Babble(BaseEnvironment, gym.Env):
    def __init__(
        self,
        path_to_logfile: Optional[str] = None,
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
        distance: int = 2.5,
        lookat: np.ndarray = np.array([0.45, 0, -0.5]),
        ) -> None:

        self.type = 'arm26babble'
        self.task = task

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
                open(path_to_logfile,'ab') if path_to_logfile else None
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
        self.muscle_strength_factor = np.array([muscle_strength_factor])
        self.set_muscle_strength_factor(self.muscle_strength_factor)

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

        # ---
        # Figures
        # ---

        # Weight matrix visualization
        W_Ia_heatmap = viz.HeatmapGeom(
            center= lookat + np.array([-0.92, -1.2, -0.10]),
            size=0.05,
            W=spinal_layer.net.aMN.W_Ia,
            xlabel=['Ia_SF', 'Ia_SE', 'Ia_EF', 'Ia_EE', 'Ia_BF', 'Ia_BE'],
            ylabel=['MN_SF', 'MN_SE', 'MN_EF', 'MN_EE', 'MN_BF', 'MN_BE']
        )
        self.add_geom(W_Ia_heatmap)

        W0 = spinal_layer.net.aMN.W_Ia.clone().detach()
        diffW_Ia_heatmap = viz.HeatmapGeom(
            center= lookat + np.array([-0.8, -2.1, -5]),
            size=0.03,
            W=spinal_layer.net.aMN.W_Ia,
            W0=W0,
            notext=True,
            spacing=2
        )
        self.add_geom(diffW_Ia_heatmap)


        # Vertical Stack of Trace Monitors
        self.figs = []
        T = 1
        H = 80;
        W = 300
        xs = 900
        ys = 900
        self.N = int(T / self.model.opt.timestep)

        # qpos
        fig = viz.TracesFigure(
            viewport=mujoco.MjrRect(xs, ys-H, W, H),
            ntraces=2,
            window=self.N,
            source=self.data.qpos,
            ylabel='qpos',
            xlim=np.array([0, T]),
            ylim=np.array([-0.25, 0.25]),
            ysymmetric=True
        )
        self.add_fig(fig)
        # qvel
        fig = viz.TracesFigure(
            viewport=mujoco.MjrRect(xs, ys-2*H, W, H),
            ntraces=2,
            window=self.N,
            source=self.data.qvel,
            ylabel='qvel',
            xlim=np.array([0, T]),
            ylim=np.array([-0.1, 0.1]),
            ysymmetric=True
        )
        self.add_fig(fig)
        # length
        fig = viz.TracesFigure(
            viewport=mujoco.MjrRect(xs, ys-3*H, W, H),
            ntraces=6,
            window=self.N,
            source=self.spinal_layer.length,
            ylabel='len',
            xlim=np.array([0, T]),
            ylim=np.array([0, 1])
        )
        self.add_fig(fig)
        # velocity
        fig = viz.TracesFigure(
            viewport=mujoco.MjrRect(xs, ys-4*H, W, H),
            ntraces=6,
            window=self.N,
            source=self.spinal_layer.velocity,
            ylabel='vel',
            xlim=np.array([0, T]),
            ylim=np.array([-0.1, 0.1]),
            ysymmetric=True
        )
        self.add_fig(fig)
        # force
        fig = viz.TracesFigure(
            viewport=mujoco.MjrRect(xs, ys-5*H, W, H),
            ntraces=6,
            window=self.N,
            source=self.spinal_layer.force,
            ylabel='force',
            xlim=np.array([0, T]),
            ylim=np.array([-1, 0])
        )
        self.add_fig(fig)
        # act_II
        fig = viz.TracesFigure(
            viewport=mujoco.MjrRect(xs, ys-6*H, W, H),
            ntraces=6,
            window=self.N,
            source=self.spinal_layer.act_II,
            ylabel='act_II',
            xlim=np.array([0, T]),
            ylim=np.array([0, 1]),
            gridsize=np.array([2, 3])
        )
        self.add_fig(fig)
        # act_Ia
        fig = viz.TracesFigure(
            viewport=mujoco.MjrRect(xs, ys-7*H, W, H),
            ntraces=6,
            window=self.N,
            source=self.spinal_layer.act_Ia,
            ylabel='act_Ia',
            xlim=np.array([0, T]),
            ylim=np.array([0, 1]),
            gridsize=np.array([2, 3])
        )
        self.add_fig(fig)
        # babble_signal
        fig = viz.TracesFigure(
            viewport=mujoco.MjrRect(xs, ys-8*H, W, H),
            ntraces=6,
            window=self.N,
            source=self.babble_signal,
            ylabel='babble',
            xlim=np.array([0, T]),
            ylim=np.array([0, 1]),
            gridsize=np.array([2, 3])
        )
        self.add_fig(fig)
        # act_aMN
        fig = viz.TracesFigure(
            viewport=mujoco.MjrRect(xs, ys-9*H, W, H),
            ntraces=6,
            window=self.N,
            source=self.spinal_layer.act_aMN,
            ylabel='MN',
            xlim=np.array([0, T]),
            ylim=np.array([0, 1]),
            gridsize=np.array([2, 3])
            )
        self.add_fig(fig)
        # act_muscle
        fig = viz.TracesFigure(
            viewport=mujoco.MjrRect(xs, ys-10*H, W, H),
            ntraces=6,
            window=self.N,
            source=self.spinal_layer.act_muscle,
            ylabel='act_MN',
            xlim=np.array([0, T]),
            ylim=np.array([0, 1]),
            gridsize=np.array([2, 3])
        )
        self.add_fig(fig)
        # activation
        fig = viz.TracesFigure(
            viewport=mujoco.MjrRect(xs, ys-11*H, W, H),
            ntraces=6,
            window=self.N,
            source=self.spinal_layer.activation,
            ylabel='act',
            xlim=np.array([0, T]),
            ylim=np.array([0, 1]),
            gridsize=np.array([2, 3])
        )
        self.add_fig(fig)

        #xlabel
        fig = viz.TracesFigure(
            viewport=mujoco.MjrRect(xs, ys-11*H - 20, W, 10),
            ntraces=0,
            window=self.N,
            xlim=np.array([-T, 0])
        )
        fig.figure.yformat = ''
        fig.figure.flg_ticklabel[0] = 1
        fig.figure.flg_ticklabel[1] = 1
        self.add_fig(fig)

        # Muscle Strength Factor
        fig = viz.BarsFigure(
            viewport=mujoco.MjrRect(220, 0, 150, 100),
            nbars=1,
            title=f'Muscle Strength',
            source=self.muscle_strength_factor,
            ylim=np.array([-1, 1]),
            xlim=np.array([0, 1]),
            gridsize=np.array([5, 2]),
            ysymmetric=True
        )
        fig.figure.flg_selection = 1
        fig.figure.selection = self.muscle_strength_factor
        self.add_fig(fig)

    def set_muscle_strength_factor(self, factor):
        self.muscle_strength_factor[:] = factor
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
        self.babble_signal[:] = np.array([
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
            self._render()
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
            self.logfile = open(path_to_logfile, 'ab')
        else:
            self.logfile = None

    def _update_figures(self):
        for fig in self.figs:
            fig.update()

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
            # As a string
            #data_str = ','.join(map('{:.6f}'.format, data))
            #print(data_str, file=self.logfile)

            # As a byte string
            self.logfile.write(data.tobytes())

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


