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
        ic_type: Optional[str] = 'joints_only',
        reward_type: Optional[str] = 'simple',
        max_time = 30,
        min_time = 10,
        qmax = np.pi/8*np.ones(2,),
        pmax = np.pi/4,
        spine: Optional[Any] = None,
        supraspine: Optional[Any] = None,
        damping = 1,
        viscosity = 0,
        gravity = [9.81, 0, 0],
        body_mass: np.ndarray = [0, 2, 1, 1, 0],
        pendulum_length: float = None,
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

        # Update body masses and pendulum length via MjSpec, then
        # recompile so MuJoCo recomputes inertia/ipos exactly.
        # Must happen BEFORE spine/supraspine init (they cache model refs).
        for geom_spec in self.spec.geoms:
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_spec.name
            )
            if gid >= 0:
                bid = self.model.geom_bodyid[gid]
                new_mass = body_mass[bid]
                if new_mass > 0 and geom_spec.mass > 0:
                    geom_spec.mass = float(new_mass)

        if pendulum_length is not None:
            pend = self.spec.geom('pendulum')
            pend.fromto = [0, 0, 0, pendulum_length, 0, 0]
            pend_end_body = list(self.spec.bodies)[4]
            pend_end_body.pos = [pendulum_length, 0, 0]

        self.model, self.data = self.spec.recompile(self.model, self.data)

        # Apply non-geom mechanical properties (after recompile)
        self.model.opt.gravity = gravity
        self.model.opt.viscosity = viscosity
        self.model.dof_damping[:] = damping
        mujoco.mj_forward(self.model, self.data)

        # Now initialize spine/supraspine with the final model
        self.spine = self.initialize_spine(spine)
        self.supraspine = self.initialize_supraspine(supraspine)

        self.task = task
        _valid_ic_types = ('pend_only', 'joints_only')
        if ic_type not in _valid_ic_types:
            raise ValueError(f"ic_type={ic_type!r} not in {_valid_ic_types}")
        self.ic_type = ic_type
        _valid_reward_types = ('simple', 'simple-gauss', 'log', 'integral',
                               'integral-gauss', 'gauss',
                               'vel-angle', 'simple-vel-angle')
        if reward_type not in _valid_reward_types:
            raise ValueError(f"reward_type={reward_type!r} not in {_valid_reward_types}")
        self.reward_type = reward_type
        self.qmax = qmax
        self.qmin = -self.qmax
        self.pmax = pmax

        self.done = False
        self.terminated = False
        self.truncated = False

        self.logging_enabled = True
        self.eps = 1
        self.reward = 0
        self.seed = seed
        self.solved = 0

        # IC logging
        self.current_ic_q0 = 0.0
        self.current_ic_q1 = 0.0
        self.current_ic_pend = 0.0
        self.log_ic_q0 = 0.0
        self.log_ic_q1 = 0.0
        self.log_ic_pend = 0.0

        # Final condition logging
        self.log_fc_q0 = 0.0
        self.log_fc_q1 = 0.0
        self.log_fc_q2 = 0.0
        self.log_fc_pend = 0.0

        self.min_time = min_time
        self.max_time = max_time

        np.random.seed(self.seed)
        self.ic_rng = np.random.default_rng(self.seed + 1)

        if relative_muscle_scale is None:
            relative_muscle_scale = np.ones(self.model.nu)
        self.relative_muscle_scale = relative_muscle_scale

        # Scale the muscle's peak force F0 (gainprm[:,2]) directly, NOT `scale`
        # (gainprm[:,3]), as in arm26_babble.py. The XML sets an explicit
        # force>=0, so MuJoCo ignores `scale` and no longer auto-sizes
        # F0 = scale/acc0 -- keeping F0 decoupled from limb inertia (body_mass).
        # biasprm[:,2] carries F0 for the passive force and is scaled with it.
        self.default_muscle_gain = (
                relative_muscle_scale * self.model.actuator_gainprm[:,2]
        )
        self.default_muscle_bias = (
                relative_muscle_scale * self.model.actuator_biasprm[:,2]
        )
        self.muscle_strength = np.array([muscle_strength])
        self.set_muscle_strength(self.muscle_strength)

        # Cache for the tb log file
        self.log_eps = 0
        self.ep_reward = 0
        self.log_eps_reward = 0

        np.set_printoptions(precision=2, suppress=True)
        print('\n--- Environment Properties ---')
        print(f'Model: {xml_path}')
        print(f'Reward type: {self.reward_type}')
        print(f'IC type:     {self.ic_type}')
        print(f'pmax:        {self.pmax:.4f} rad ({np.degrees(self.pmax):.1f} deg)')
        print(f'qmax:        {self.qmax}  (qmin = -qmax)')
        print(f'min_time:    {self.min_time} s')
        print(f'max_time:    {self.max_time} s')
        print('body_mass: ', self.model.body_mass)
        print('body inertia: ', self.model.body_inertia)
        print('pendulum length: ', pendulum_length)
        print(f'gravity: {self.model.opt.gravity}')
        print(f'viscosity: {self.model.opt.viscosity}')
        print(f'damping: {self.model.dof_damping}')
        print(f'muscle_strength: {self.muscle_strength}\n')
        print('Muscle Normalization:')
        print('range: \n', self.model.actuator_gainprm[:, 0:2])
        print('force: ', self.model.actuator_gainprm[:, 2])   # peak active force F0 (N)
        print('L0: ', self.spine.L0)
        print('lmin: ', self.spine.lmin)
        print('lmax: ', self.spine.lmax)
        print('vmax: ', self.spine.vmax)
        self.spine._sample_mechanics(self.data)
        print('normalized length at rest: ', self.spine.length)
        print()
        np.set_printoptions()

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
        self.model.actuator_gainprm[:,2] = (
            strength * self.default_muscle_gain
        )
        self.model.actuator_biasprm[:,2] = (
            strength * self.default_muscle_bias
        )


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
        if not self.logging_enabled:
            return
        self.log_eps += 1
        self.log_eps_len = self.data.time
        self.log_eps_reward = self.ep_reward
        self.log_ic_q0 = self.current_ic_q0
        self.log_ic_q1 = self.current_ic_q1
        self.log_ic_pend = self.current_ic_pend
        self.log_fc_q0 = self.data.qpos[0]
        self.log_fc_q1 = self.data.qpos[1]
        self.log_fc_pend = np.sum(self.data.qpos) % (2*np.pi) - np.pi

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
            self.inverted_flag = True
            if self.ic_type == 'pend_only':
                self.data.qpos[0] = 0.0
                self.data.qpos[1] = 0.0
                self.data.qpos[2] = (
                    self.ic_rng.uniform(-self.pmax,self.pmax) - np.pi
                )
            elif self.ic_type == 'joints_only':
                self.data.qpos[0] = (
                    self.ic_rng.uniform(self.qmin[0],self.qmax[0])
                )
                self.data.qpos[1] = (
                    self.ic_rng.uniform(self.qmin[1],self.qmax[1])
                )
                self.data.qpos[2] = (
                    np.pi - self.data.qpos[0] - self.data.qpos[1]
                )
        else:
            self.data.qpos[0:3] = np.zeros(3,)
            self.data.qvel = np.zeros(3,)

        self.pendulum_pos = np.sum(self.data.qpos) % (2*np.pi) - np.pi

        # Cache IC for logging
        self.current_ic_q0 = self.data.qpos[0]
        self.current_ic_q1 = self.data.qpos[1]
        self.current_ic_pend = self.pendulum_pos

        mujoco.mj_forward(self.model, self.data)

        self.obs = self.supraspine.sample()

        self.simulation_steps = 0

    def _update_status(self):

        if self.done:
            return

        # Entering inverted state (0 = balanced, ±pi = hanging)
        if (not self.inverted_flag
                and abs(self.pendulum_pos) <= self.pmax):
          self.inverted_flag = True
          self.t0 = self.data.time

        # Check fail or solved
        if self.inverted_flag:
            out_of_bounds = abs(self.pendulum_pos) > self.pmax
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

        mu = 0
        sig = np.pi/8
        scale = 1
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
            rew = self.inverted_flag - (not self.inverted_flag)
        elif self.reward_type == 'simple-gauss':
            rew = (
                np.exp(-0.5 * ((self.pendulum_pos - mu)/sig)**2) / scale
                + (self.inverted_flag - (not self.inverted_flag))
            )
        elif self.reward_type == 'gauss':
            rew = np.exp(-0.5 * ((self.pendulum_pos - mu)/sig)**2) / scale
        elif self.reward_type == 'integral-gauss':
            rew = (
                np.exp(-0.5 * ((self.pendulum_pos - mu)/sig)**2) / scale
                + self.inverted_flag * (self.data.time - self.t0)
            )
        elif self.reward_type == 'log':
            rew = self.inverted_flag*(np.log(1 + (self.data.time - self.t0)))
        elif self.reward_type == 'integral':
            rew = self.inverted_flag*(self.data.time - self.t0)
        elif self.reward_type == 'vel-angle':
            gate     = float(self.pendulum_vel * self.pendulum_pos <= 0)
            rew = (1 - abs(self.pendulum_pos) / np.pi) * gate
        elif self.reward_type == 'simple-vel-angle':
            gate     = float(self.pendulum_vel * self.pendulum_pos <= 0)
            rew = (
                (1 - abs(self.pendulum_pos) / np.pi) * gate
                + (self.inverted_flag - (not self.inverted_flag))
            )
        self.reward += rew/self.simulation_sampling_rate


    def step(self, action):
        self.reward = 0.0

        # Apply most recent action from supraspine
        self.spine.set_descending_drive(self.supraspine.sample_action())

        # Update action
        self.supraspine.set_action(action)

        # Step physics forward decimated number of loops
        for steps in range(self.supraspine.decimation_factor):
            mujoco.mj_step(self.model, self.data)

            self.pendulum_pos = np.sum(self.data.qpos) % (2*np.pi) - np.pi
            self.pendulum_vel = np.sum(self.data.qvel)

            self._render()
            self._update_figures()
            self._update_status()
            self._update_reward()

            self.supraspine.update()
            self._write_data_to_logfile()

            self.simulation_steps += 1

            if self.done:
                break

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
                n.array([self.eps]),
                np.array([self.data.time]),
                self.data.qpos,
                self.data.qvel,
                self.spine.length,
                self.spine.velocity,
                self.spine.force,
                self.spine.act_II,
                self.spine.act_Ia,
                self.spine.act_MN,
                self.spine.drive,
                self.spine.act_muscle,
                self.spine.act,
                self.supraspine.visual,
                self.supraspine.action,
            ])
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
            self.logger.record("eps/ep_len", self.env.log_eps_len)
            self.logger.record("eps/eps", self.env.log_eps)
            self.logger.record("eps/solved", self.env.solved)
            self.logger.record("eps/ic_q0", self.env.log_ic_q0)
            self.logger.record("eps/ic_q1", self.env.log_ic_q1)
            self.logger.record("eps/ic_pend", self.env.log_ic_pend)
            self.logger.record("eps/fc_q0", self.env.log_fc_q0)
            self.logger.record("eps/fc_q1", self.env.log_fc_q1)
            self.logger.record("eps/fc_pend", self.env.log_fc_pend)

            #reset
            self.log_eps_reward = 0
            self.logger.dump(self.num_timesteps)

        return True



