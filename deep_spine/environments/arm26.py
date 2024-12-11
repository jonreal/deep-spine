from typing import Optional, Any
import gymnasium as gym
import numpy as np
import mujoco as mj

from stable_baselines3.common.callbacks import BaseCallback
from deep_spine.environments.base_environment import BaseEnvironment

class Arm26(BaseEnvironment, gym.Env):
    def __init__(
        self,
        path_to_logfile: Optional[str] = None,
        simulation_sampling_rate: int = 200,
        task: Optional[str] = 'static-pose',
        qd: np.ndarray = np.array([np.pi/4, np.pi/4]),
        episode_length: int = 10,
        max_noise_sigma: float = 0.0,
        seed: Optional[int] = 12345,
        spinal_layer: Optional[Any] = None,
        supraspinal_decimation_factor: int = 1,
        actuator_penalty: int = 0,
        velocity_penalty: int = 0,
        qd0range: np.ndarray = np.array([-np.pi/25, np.pi/2]),
        qd1range: np.ndarray = np.array([np.pi/50, np.pi/2]),
        xml_path: str = 'arm26_w_ghost.xml',
        frame_rate: Optional[int] = None,
        save_frames_to_path: Optional[str] = None,
        azimuth: int = 180,
        elevation: int = -90,
        distance: int = 2,
        lookat: np.ndarray = np.array([0.35, 0, 0.25]),
        ) -> None:

        self.type = 'arm26'
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
        self.max_noise_sigma = max_noise_sigma
        self.actuator_penalty = actuator_penalty
        self.velocity_penalty = velocity_penalty

        self.qd0range = qd0range
        self.qd1range = qd1range
        self.qd = qd
        self.q_rest = np.array([-0.00309053,  0.03906269])

        self.done = False
        self.target = np.zeros(4,)
        self.noise = np.zeros(2,)
        self.eps = 0
        self.reward = 0
        self.eps_reward = 0
        self.seed = seed
        self.ic_rng = np.random.default_rng(self.seed)

        # Cache for the log file
        self.log_eps = 0
        self.log_eps_reward = 0
        self.log_target = np.zeros(4,)

        self.spinal_layer = self.initialize_spinal_layer(spinal_layer)

        supraspinal_afferent_space = gym.spaces.Box(
            low=np.array([-np.pi,-np.pi,0,0]),
            high=np.array([np.pi,np.pi,1,1])
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

    def initialize_spinal_layer(self, spinal_layer):
        spinal_layer.initialize_motor_units(
            self.model.ntendon, self.model.actuator_lengthrange
        )
        self.action_space = spinal_layer.efferent_space
        mj.set_mjcb_control(spinal_layer.callback)
        return spinal_layer

    def _update_log(self) -> None:
        """ Update the cache for the log file """
        self.log_eps_reward = self.eps_reward
        self.log_eps = self.eps
        self.log_target = self.target

    def _sample_afferent_signals(self):
        self.afferent_signals = np.concatenate(
            (self.spinal_layer.afferent_signals, self.target)
        )

    def _update_ghost(self):
        self.data.mocap_pos[0:2] = self.data.xpos[1:3]
        self.data.mocap_quat[0:2] = self.data.xquat[1:3]

    def _update_reward(self):
        p1 = (
            -np.linalg.norm(self.target[0:2] - self.data.qpos[0:2], ord=2)
        )
        p2 = (
            -self.actuator_penalty
                * np.linalg.norm(self.spinal_layer.activations, ord=2)
        )
        p3 = (
            -self.velocity_penalty
                * np.linalg.norm(self.data.qvel, ord=2)
        )
        self.reward += (
            ((p1 + p2 + p3) / self.simulation_sampling_rate)
            # / (
            #    (1 + np.linalg.norm(self.q_rest - self.target[3:], ord=2))
            #   * (1 + np.linalg.norm(self.noise, ord=2))
            #)
        )

    def _set_initial_conditions(self):
        self.done = False
        self.reward = 0
        self.eps_reward = 0
        mj.mj_resetData(self.model, self.data)
        self.spinal_layer.set_efferent_signals(
            np.zeros(self.spinal_layer.n,)
        )
        self.target = self._gen_target()
        self.data.qpos[0] = self.target[0]
        self.data.qpos[1] = self.target[1]
        self.noise[0] = self.target[2]
        self.noise[1] = self.target[3]
        self._update_joint_noise()
        mj.mj_forward(self.model, self.data)
        self._update_ghost()

        large_prime = 78531
        self.seed = str(int(self.seed * large_prime * (self.eps + 1)))
        self.seed = int(self.seed[:5])
        self.ic_rng = np.random.default_rng(self.seed)
        self.simulation_steps = 0

    def _gen_target(self):
        if self.task == 'static-pose':
            return np.append(
                    self.qd,
                    np.array([self.ic_rng.uniform(0,self.max_noise_sigma),
                        self.ic_rng.uniform(0,self.max_noise_sigma)]
                    )
            )
        elif self.task == 'random-static-pose':
            return (
                np.array(
                    [self.ic_rng.uniform(self.qd0range[0],self.qd0range[1]),
                    self.ic_rng.uniform(self.qd1range[0],self.qd1range[1]),
                    self.ic_rng.uniform(0,self.max_noise_sigma),
                    self.ic_rng.uniform(0,self.max_noise_sigma)]
                )
            )
            raise ValueError(f"Invalid task: {self.task}")

    def _update_status(self):
        if self.simulation_steps >= self.simulation_steps_per_episode:
            self.done = True
            self.eps += 1

    def _update_joint_noise(self):
        self.data.ctrl[-2] = np.random.normal(0,self.noise[0])
        self.data.ctrl[-1] = np.random.normal(0,self.noise[1])

    def step(self, action):
        self.reward = 0
        self.spinal_layer.set_efferent_signals(action)
        for steps in range(self.supraspinal_decimation_factor):
            mj.mj_step(self.model, self.data)
            self.simulation_steps += 1
            self._render()
            self._sample_afferent_signals()
            self._update_reward()
            self._update_status()
            self._write_data_to_log()
            self._update_joint_noise()

        self.eps_reward += self.reward
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
            arrays = [
                self.data.qpos, self.data.qvel, self.afferent_signals,
                self.spinal_layer.efferent_signals
            ]
            data_str = ','.join(map(str, np.concatenate(arrays)))
            print(data_str, file=self.log)

class LogCallbackArm26(BaseCallback):
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
            self.logger.record(
                "eps/ep_rew", self.env.log_eps_reward)
            self.logger.record(f"eps/eps",self.env.log_eps)
            self.logger.record(f"eps/q1d",self.env.log_target[0])
            self.logger.record(f"eps/q2d",self.env.log_target[1])
            self.logger.record(f"eps/n1",self.env.log_target[2])
            self.logger.record(f"eps/n2",self.env.log_target[3])
            self.logger.dump(self.num_timesteps)
        return True


