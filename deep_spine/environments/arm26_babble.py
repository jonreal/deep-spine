from typing import Optional, Any
import gymnasium as gym
import numpy as np
import mujoco
from stable_baselines3.common.callbacks import BaseCallback
import gzip

from deep_spine.environments.base_environment import BaseEnvironment
import deep_spine.utils.viz as viz
from deep_spine.spine.filters import iir

class Arm26Babble(BaseEnvironment, gym.Env):
    def __init__(
        self,
        path_to_logfile: Optional[str] = None,
        task: Optional[str] = '',
        spine: Optional[Any] = None,
        supraspine: Optional[Any] = None,
        muscle_strength: float = 1,
        episode_length: int = 100,
        seed: Optional[int] = 12345,
        xml_path: str = 'arm26.xml',
        frame_rate: Optional[int] = None,
        save_frames_to_path: Optional[str] = None,
        render_in_background: bool = False,
        render_lossless: bool = False,
        window_width: int = 1200,
        window_height: int = 900,
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
            window_width=window_width,
            window_height=window_height,
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

        self.spine = self.initialize_spine(spine)
        self.supraspine = self.initialize_supraspine(supraspine)

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

        self.max_muscle_strengths = self.model.actuator_gainprm[:,3].copy()
        self.muscle_strength = np.array([muscle_strength])
        self.set_muscle_strength(self.muscle_strength)

        # Cache for the tb log file
        self.log_eps = 0
        self.log_eps_reward = 0


    def set_muscle_strength(self, strength):
        self.muscle_strength[:] = strength
        self.model.actuator_gainprm[:,3] = strength * self.max_muscle_strengths

    def initialize_spine(self, spine):
        spine.initialize(self.model, self.data)
        mujoco.mj_resetData(self.model,self.data)
        mujoco.mj_forward(self.model, self.data)
        self.simulation_sampling_rate = 1 / self.model.opt.timestep
        mujoco.set_mjcb_control(spine.callback)
        return spine

    def initialize_supraspine(self, supraspine):
        supraspine.initialize(self.model, self.data)
        return supraspine

    def _set_initial_conditions(self):
        self.done = False
        #mujoco.mj_resetData(self.model, self.data)
        self.data.time = 0
        self.spine.set_descending_drive(np.zeros(self.spine.n,))
        #mujoco.mj_forward(self.model, self.data)

        large_prime = 78531
        self.seed = str(int(self.seed * large_prime * (self.eps + 1)))
        self.seed = int(self.seed[:5])
        self.ic_rng = np.random.default_rng(self.seed)
        self.supraspine.update_seed(self.seed)
        self.simulation_steps = 0

#    def _gen_babble(self):
#        if self.task == 'babble':
#            for i in range(self.spinal_layer.n):
#                if (self.data.time
#                    >= self.babble_duration[i] + self.babble_t0[i]
#                ) or (
#                    self.babble_prob[i] == 0
#                ) or (
#                    self.data.time == 0
#                ):
#                    self.babble_prob[i] = (
#                        self.ic_rng.uniform(0, 1) <= self.babble_max_prob
#                    )
#                    if self.babble_prob[i]:
#                        self.babble_t0[i] = self.data.time
#                        self.babble_amp[i] = (
#                            self.ic_rng.uniform(0, 1)
#                        )
#                        self.babble_duration[i] = self.ic_rng.uniform(
#                            self.babble_duration_range[0],
#                            self.babble_duration_range[1]
#                        )
#                        self.babble_rise_time[i] = self.ic_rng.uniform(
#                            self.babble_rise_time_range[0],
#                            self.babble_rise_time_range[1]
#                        )
#                        self.babble_filter[i] = iir.IirFilt(
#                                [0, self.babble_rise_time[i]],
#                                [1, self.babble_rise_time[i] - 1]
#                        )
#        else:
#            raise ValueError(f"Invalid task: {self.task}")
#
#    def _update_babble(self):
#        self.babble_signal[:] = np.array([
#            self.babble_prob[i]
#            * self.babble_amp[i]
#            * self.babble_filter[i].filter(
#                self.data.time <= self.babble_duration[i] + self.babble_t0[i]
#            ) for i in range(self.spinal_layer.n)
#        ])


    def _update_status(self):
        if self.simulation_steps >= self.simulation_steps_per_episode:
            self.done = True
            self.eps += 1

    def step(self, action):
        self.reward = 0
        #self.spinal_layer.set_descending_drive(self.babble_signal)

        drive = self.supraspine.forward(self.data);
        self.spine.set_descending_drive(drive)

        for steps in range(self.supraspine.decimation_factor):
            mujoco.mj_step(self.model, self.data)
            self.simulation_steps += 1
            self._update_figures()
            self._render()
            self._update_status()
            self._write_data_to_logfile()
            #self._update_babble()
            #self._gen_babble()

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
                self.spine.length,
                self.spine.velocity,
                self.spine.force,
                self.spine.act_II,
                self.spine.act_Ia,
                self.supraspine.drive,
                self.spine.act_aMN,
                self.spine.act_muscle,
                self.spine.activation,
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


