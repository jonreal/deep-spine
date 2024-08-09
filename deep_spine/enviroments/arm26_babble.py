from typing import Optional, Any
import gymnasium as gym
import numpy as np
import mujoco as mj

from stable_baselines3.common.callbacks import BaseCallback
from deep_spine.enviroments.base_environment import BaseEnvironment
from deep_spine.spinal_layers.filters import iir

class Arm26Babble(BaseEnvironment, gym.Env):
    def __init__(
        self,
        path_to_logfile: Optional[str] = None,
        simulation_sampling_rate: int = 200,
        task: Optional[str] = 'babble',
        xcor_window: int = 10,
        episode_length: int = 100,
        seed: Optional[int] = 12345,
        spinal_layer: Optional[Any] = None,
        supraspinal_decimation_factor: int = 1,
        xml_path: str = 'arm26.xml',
        frame_rate: Optional[int] = None,
        save_frames_to_path: Optional[str] = None,
        azimuth: int = 180,
        elevation: int = -90,
        distance: int = 2,
        lookat: np.ndarray = np.array([0.35, 0, 0.25]),
        ) -> None:

        self.type = 'arm26-babble'
        self.task = task
        self.simulation_sampling_rate = simulation_sampling_rate

        super().__init__(
            xml_path=xml_path,
            frame_rate=frame_rate,
            save_frames_to_path=save_frames_to_path,
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

        self.episode_length = episode_length
        self.simulation_steps_per_episode = (
            self.episode_length * simulation_sampling_rate
        )

        self.done = False
        self.eps = 0
        self.reward = 0
        self.eps_reward = 0
        self.seed = seed
        self.ic_rng = np.random.default_rng(self.seed)

        self.spinal_layer = self.initialize_spinal_layer(spinal_layer)


        # Actions: [activation_MN (spinal.efferent), activation_Ia]

        # This is for activation_Ia, which is not used by 'low-level' spinal
        supraspinal_effector_space = gym.spaces.Box(
            low=np.zeros(self.spinal_layer.n,),
            high=np.ones(self.spinal_layer.n,)
        )
        self.action_space = gym.spaces.Box(
            low=np.concatenate(
                [spinal_layer.efferent_space.low,
                supraspinal_effector_space.low]
            ),
            high=np.concatenate(
                [spinal_layer.efferent_space.high,
                supraspinal_effector_space.high]
            )
        )

        # Features: [spinal_afferents, supraspinal_afferents]
        # [length, velocity, a_MN, a_babble]
        supraspinal_afferent_space = gym.spaces.Box(
            low=np.zeros(2*self.spinal_layer.n,),
            high=np.ones(2*self.spinal_layer.n,)
        )
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                [spinal_layer.afferent_space.low,
                 supraspinal_afferent_space.low]
            ),
            high=np.concatenate(
                [spinal_layer.afferent_space.high,
                 supraspinal_afferent_space.high]
            )
        )
        self.supraspinal_decimation_factor = supraspinal_decimation_factor

        self.xcor_window = xcor_window
        self.xcorfilts = self._initialize_filters()

        self.activation_MN = np.zeros(self.spinal_layer.n,)
        self.activation_Ia = np.zeros(self.spinal_layer.n,)
        self.babble = np.zeros(self.spinal_layer.n,)

        # Cache for the log file
        self.log_eps = 0
        self.log_eps_reward = 0
        self.log_babble = self.babble


    def initialize_spinal_layer(self, spinal_layer):
        spinal_layer.initialize_motor_units(
            self.model.ntendon, self.model.actuator_lengthrange
        )
        mj.set_mjcb_control(spinal_layer.callback)
        return spinal_layer

    def _initialize_filters(self):
        b = np.zeros(self.xcor_window,)
        a = np.zeros(self.xcor_window,)
        print(self.xcor_window)
        a[0] = 1
        a[1] = -1
        b[0] = 1
        b[self.xcor_window - 1] = -1
        filters = [
            iir.IirFilt(b,a)
            for _ in range(self.spinal_layer.n**2)
        ]
        return filters


    def _update_log(self) -> None:
        """ Update the cache for the log file """
        self.log_eps_reward = self.eps_reward
        self.log_eps = self.eps
        self.log_babble = self.babble


    # Bad name for this function
    # sample_obs
    def _sample_afferent_signals(self):
        self.afferent_signals = np.concatenate(
            (
                self.spinal_layer.afferent_signals,
                self.activation_MN,
                self.activation_Ia,
            )
        )

    def _update_reward(self):
        outer_product = np.outer(
            self.activation_MN,
            self.activation_Ia
        )
        self.correlation = (
            [f.filter(x)
            for f, x  in zip(
                self.xcorfilts,
                outer_product.flatten() / (self.spinal_layer.n**2)
            )]
        )
        self.reward = np.linalg.norm(self.correlation)

    def _set_initial_conditions(self):
        self.done = False
        self.reward = 0
        self.eps_reward = 0
        mj.mj_resetData(self.model, self.data)
        self.spinal_layer.set_efferent_signals(
            np.zeros(self.spinal_layer.n,)
        )
        self.babble = self._gen_babble()
        mj.mj_forward(self.model, self.data)

        large_prime = 78531
        self.seed = str(int(self.seed * large_prime * (self.eps + 1)))
        self.seed = int(self.seed[:5])
        self.ic_rng = np.random.default_rng(self.seed)
        self.simulation_steps = 0

    def _gen_babble(self):
        if self.task == 'babble':
            return np.array(
                [self.ic_rng.uniform(low, high) for low, high in zip(
                    np.zeros(self.spinal_layer.n,),
                    np.ones(self.spinal_layer.n,)
                )]
            )
        else:
            raise ValueError(f"Invalid task: {self.task}")

    def _update_babble(self):
        self.activation_babble = (
            np.array(
                [babble + self.ic_rng.normal(0, 0.5)
                for _, babble in enumerate(self.babble)]
            )
        )

    def _update_status(self):
        if self.simulation_steps >= self.simulation_steps_per_episode:
            self.done = True
            self.eps += 1

    def step(self, action):
        self.reward = 0

        self.activation_MN = action[:self.spinal_layer.n]
        self.activation_Ia = action[self.spinal_layer.n:]

        self.spinal_layer.set_efferent_signals(self.activation_MN)
        for steps in range(self.supraspinal_decimation_factor):
            mj.mj_step(self.model, self.data)
            self.simulation_steps += 1
            self._render()
            self._sample_afferent_signals()
            self._update_reward()
            self._update_status()
            self._write_data_to_log()
            self._update_babble()
        self.eps_reward += (self.reward / self.simulation_sampling_rate)
        truncated = False
        info = {}
        return self.afferent_signals, self.reward, self.done, truncated, info

    def reset(self,seed=None,**kwargs):
        super().reset(seed=seed)
        self._update_log()
        self.done = False
        self.count = 0
        self._set_initial_conditions()
        self._sample_afferent_signals()
        info = {}
        return self.afferent_signals, info

    def _write_data_to_log(self):
        if self.logfile:
            data_str = (
                ', '.join(map(str, self.data.qpos)) + ', '
                + ', '.join(map(str, self.data.qvel)) + ', '
                + ', '.join(map(str, self.afferent_signals)) + ', '
                + ', '.join(map(str, self.spinal_layer.activations))
            )
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
            self.logger.record(f"eps/b1",self.env.log_babble[0])
            self.logger.record(f"eps/b2",self.env.log_babble[1])
            self.logger.record(f"eps/b3",self.env.log_babble[2])
            self.logger.record(f"eps/b4",self.env.log_babble[3])
            self.logger.record(f"eps/b5",self.env.log_babble[4])
            self.logger.record(f"eps/b6",self.env.log_babble[5])
            self.logger.dump(self.num_timesteps)
        return True


