from typing import Optional, Any
import gymnasium as gym
import numpy as np
import mujoco
from stable_baselines3.common.callbacks import BaseCallback
import gzip

import torch

from deep_spine.environments.base_environment import BaseEnvironment
import deep_spine.utils.viz as viz
from deep_spine.spine.filters import iir

def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class Arm26Pendulum(BaseEnvironment, gym.Env):
    def __init__(
        self,
        path_to_log: Optional[str] = None,
        task: Optional[str] = 'balance',
        reward_type: Optional[str] = 'simple',
        max_time = 300,
        min_time = 10,
        q0min = -np.pi/8*np.ones(2,),
        q0max = np.pi/8*np.ones(2,),
        spine: Optional[Any] = None,
        supraspine: Optional[Any] = None,
        damping = 1,
        viscosity = 0,
        gravity = [9.81, 0, 0],
        body_mass: np.ndarray = [0, 2, 1, 0.25, 0],
        muscle_strength: float = 1,
        relative_muscle_scale: np.ndarray = None,
        seed: Optional[int] = 12345,
        xml_path: str = 'arm26_pendulum.xml',
        frame_rate: Optional[int] = None,
        save_frames_to_path: Optional[str] = None,
        render_lossless: bool = False,
        video_scale: float = 0.5,
        video_crf: str = '23',
        window_width: int = 1200,
        window_height: int = 900,
        azimuth: int = 180,
        elevation: int = -90,
        distance: int = 2.5,
        lookat: np.ndarray = np.array([0.45, 0, -0.5]),
        ) -> None:

        self.type = 'arm26pendulum'
        self.task = task
        super().__init__(
            xml_path=xml_path,
            frame_rate=frame_rate,
            save_frames_to_path=save_frames_to_path,
            render_lossless=render_lossless,
            video_scale=video_scale,
            video_crf=video_crf,
            window_width=window_width,
            window_height=window_height,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            lookat=lookat
        )
        if path_to_log:
            self.logfile = (
                open(path_to_log,'ab') if path_to_log else None
            )
        else:
            self.logfile = None

        self.spine = self.initialize_spine(spine)
        self.supraspine = self.initialize_supraspine(supraspine)

        self.task = task
        self.reward_type = reward_type
        self.q0min = q0min
        self.q0max = q0max

        self.done = False
        self.terminated = False
        self.truncated = False

        self.eps = 1
        self.reward = 0
        self.seed = seed
        self.solved = 0

        # IC logging
        self.current_ic_q0 = 0.0
        self.current_ic_q1 = 0.0
        self.log_ic_q0 = 0.0
        self.log_ic_q1 = 0.0

        # Final condition logging
        self.log_fc_q0 = 0.0
        self.log_fc_q1 = 0.0
        self.log_fc_pend = 0.0

        self.min_time = min_time
        self.max_time = max_time

        np.random.seed(self.seed)
        self.ic_rng = np.random.default_rng(self.seed + 1)

        if relative_muscle_scale is None:
            # There's a motor at the pendulum joint
            relative_muscle_scale = self.model.nu

        self.relative_muscle_scale = relative_muscle_scale

        self.default_muscle_strengths = (
                relative_muscle_scale * self.model.actuator_gainprm[:,3]
        )
        self.muscle_strength = np.array([muscle_strength])
        self.set_muscle_strength(self.muscle_strength)

        # Cache for the tb log file
        self.log_eps = 0
        self.ep_reward = 0
        self.log_eps_reward = 0

        # Mechanics Properties
        self.model.opt.gravity = gravity;
        self.model.opt.viscosity = viscosity;
        self.model.dof_damping[:] = damping


        # Scale diag inertias based on mass changes equivalent to scaling
        # density
        oldmass = self.model.body_mass
        oldinertia = self.model.body_inertia

        # s = body_mass / oldmass, but: if oldmass==0 -> leave s at 0
        eps = 0.0  # or 1e-12 if you want "near-zero counts as zero"

        s = np.divide(
            body_mass, oldmass,
            out=np.zeros_like(body_mass, dtype=float),
            where=(oldmass != eps)
        )

        self.model.body_mass = body_mass
        self.model.body_inertia = s[:, None] * oldinertia

        mujoco.mj_forward(self.model, self.data)
        print('--- Mechanical Properties ---')
        print(f'  body_mass:  {self.model.body_mass.tolist()}')
        print(f'  gravity:    {self.model.opt.gravity.tolist()}')
        print(f'  damping:    {self.model.dof_damping.tolist()}')
        print(f'  viscosity:  {self.model.opt.viscosity}')
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(self.model.nv)
        ]
        col_w = max(len(n) for n in joint_names) + 2
        header = '            ' + ''.join(n.rjust(col_w) for n in joint_names)
        moment = np.zeros((self.model.nu, self.model.nv))
        for i in range(self.model.nu):
            nnz = self.data.moment_rownnz[i]
            adr = self.data.moment_rowadr[i]
            cols = self.data.moment_colind[adr:adr+nnz]
            moment[i, cols] = self.data.actuator_moment[adr:adr+nnz]
        print(f'  moment_arms (q=0):')
        print(header)
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
            )
            vals = np.round(moment[i], 6)
            row = ''.join(f'{v:>{col_w}.4f}' for v in vals)
            print(f'    {name:>6s}  {row}')

    def update_seed(self, seed):
        self.seed = int(seed)
        np.random.seed(self.seed)
        self.ic_rng = np.random.default_rng(self.seed + 1)
        print(f'Seed: {self.seed}\n\n')

    def set_muscle_strength(self, strength):
        self.muscle_strength[:] = strength
        self.model.actuator_gainprm[:,3] = (
            strength * self.default_muscle_strengths
        )
        #print('Muscle strength set to: ', self.model.actuator_gainprm[:,3])


    def initialize_spine(self, spine):
        spine.initialize(self.model, self.data)
        mujoco.mj_resetData(self.model,self.data)
        mujoco.mj_forward(self.model, self.data)
        self.simulation_sampling_rate = 1 / self.model.opt.timestep
        mujoco.set_mjcb_control(spine.callback)
        return spine

    def initialize_supraspine(self, supraspine):
        supraspine.env = self
        supraspine.initialize(self.model, self.data)
        return supraspine

    def _update_eps_log(self):
        self.log_eps += 1
        self.log_eps_len = self.data.time
        self.log_eps_reward = self.ep_reward
        self.log_ic_q0 = self.current_ic_q0
        self.log_ic_q1 = self.current_ic_q1
        self.log_fc_q0 = self.data.qpos[0]
        self.log_fc_q1 = self.data.qpos[1]
        self.log_fc_pend = np.sum(self.data.qpos) % (2 * np.pi)

    def _set_initial_conditions(self):
        self.done = False
        self.terminated = False
        self.truncated = False
        self.inverted_flag = False
        self.reward = 0
        self.ep_reward = 0
        self.t0 = 0
        mujoco.mj_resetData(self.model, self.data)

        if self.task == 'balance':
            self.data.qpos[0] = (
                self.ic_rng.uniform(self.q0min[0],self.q0max[0])
            )
            self.data.qpos[1] = (
                self.ic_rng.uniform(self.q0min[1],self.q0max[1])
            )
            self.data.qpos[2] = np.pi - self.data.qpos[0] - self.data.qpos[1]
        else:
            self.data.qpos[0:3] = np.zeros(3,)
            self.data.qvel = np.zeros(3,)

        # Cache IC for logging
        self.current_ic_q0 = self.data.qpos[0]
        self.current_ic_q1 = self.data.qpos[1]

        mujoco.mj_forward(self.model, self.data)
        self.obs = self.supraspine.sample()

        self.simulation_steps = 0

    def _update_status(self):

        if self.done:
            return

        # Entering inverted state
        if (not self.inverted_flag and (self.pendulum_pos >= 3/4*np.pi
                 and self.pendulum_pos <= 5/4*np.pi)
          ):
          self.inverted_flag = True
          self.t0 = self.data.time

        # Check fail or solved
        if self.inverted_flag:
            out_of_bounds = (
                self.pendulum_pos < 3/4*np.pi or self.pendulum_pos > 5/4*np.pi
            )
            time_exceeded_while_inverted = (
                (self.data.time - self.t0) > self.max_time
            )

            if out_of_bounds:
                self.terminated = True
                self.eps += 1
            elif time_exceeded_while_inverted:
                self.truncated = True
                self.eps += 1
                self.solved += 1
                print("*---Solved---*\n")
        else:
            if self.data.time > self.min_time:
                self.terminated = True
                self.eps += 1

        self.done = self.terminated or self.truncated


    def _update_reward(self):

        mu = np.pi
        sig = np.pi/16
        scale = 1000
        fgauss = 0
        #if not self.done:
        #    if (self.task == 'swing-up'):
        #        fgauss = (
        #            (1/(np.sqrt(2*np.pi)*sig))*np.exp(
        #            -np.power(self.pendulum_pos-mu, 2.)/(2*np.power(sig, 2.)))
        #        )
        #        fgauss = fgauss/scale
        #else:
        #    fgauss = 0
        #fgauss = (
        #    (1/(np.sqrt(2*np.pi)*sig))*np.exp(
        #    -np.power(self.pendulum_pos-mu, 2.)/(2*np.power(sig, 2.)))
        #)
        if self.reward_type == 'simple':
            ftimeint = self.inverted_flag - (not self.inverted_flag)
        elif self.reward_type == 'log':
            ftimeint = self.inverted_flag*(np.log(1 + (self.data.time - self.t0)))
        elif self.reward_type == 'integral':
            ftimeint = self.inverted_flag*(self.data.time - self.t0)
        else:
            print(f'{self.reward_type} not found')

        self.reward += (fgauss + ftimeint)/self.simulation_sampling_rate


    def step(self, action):
        self.reward = 0.0

        # Apply most recent action from supraspine
        self.spine.set_descending_drive(self.supraspine.sample_action())

        # Update action
        self.supraspine.set_action(action)

        # Step physics forward decimated number of loops
        for steps in range(self.supraspine.decimation_factor):

            mujoco.mj_step(self.model, self.data)

            self.pendulum_pos = np.sum(self.data.qpos) % (2*np.pi)
            self.pendulum_vel = np.sum(self.data.qvel)

            self._render()
            self._update_figures()
            self._update_status()

            # Stop mid decimation if needed
            if self.done:
                break

            self._update_reward()

            self.supraspine.update()

            self.simulation_steps += 1

            #self._write_data_to_logfile()

        self.ep_reward += self.reward
        self.obs = self.supraspine.sample()

        info = {}

        return self.obs, self.reward, self.terminated, self.truncated, info

    def reset(self,seed=None,**kwargs):
        super().reset(seed=seed)
        self._update_eps_log()
        self._set_initial_conditions()
        info = {}
        return self.obs, info

    def set_logfile(self, path_to_log):
        if path_to_log:
            self.logfile = open(path_to_log, 'ab')
        else:
            self.logfile = None

    def _update_figures(self):
        if self.figs:
            for fig in self.figs:
                fig.update()

    def _write_data_to_logfile(self):
        if self.logfile:
            data = np.concatenate([
                np.array([self.data.time]),
                self.data.qpos,
                self.data.qvel,
                self.spine.length,
                self.spine.velocity,
                self.spine.force,
                self.spine.act_II,
                self.spine.act_Ia,
                self.supraspine.drive,
                self.spine.act_MN,
                self.spine.act_muscle,
                self.spine.act,
            ])
            # As a string
            #data_str = ','.join(map('{:.6f}'.format, data))
            #print(data_str, file=self.logfile)

            # As a byte string
            self.logfile.write(data.tobytes())

class LogCallbackArm26Pendulum(BaseCallback):
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
            else:
                self.logger.record(
                    f"eps/ep_rew_mean_{self.window}",
                    0
                )
            self.logger.record("eps/ep_rew", self.env.log_eps_reward)
            self.logger.record("eps/eps", self.env.log_eps)
            self.logger.record("eps/solved", self.env.solved)
            self.logger.record("eps/ic_q0", self.env.log_ic_q0)
            self.logger.record("eps/ic_q1", self.env.log_ic_q1)
            self.logger.record("eps/fc_q0", self.env.log_fc_q0)
            self.logger.record("eps/fc_q1", self.env.log_fc_q1)
            self.logger.record("eps/fc_pend", self.env.log_fc_pend)

            #reset
            self.log_eps_reward = 0
            self.logger.dump(self.num_timesteps)

        return True



