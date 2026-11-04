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
        path_to_log: Optional[str] = None,
        task: Optional[str] = '',
        spine: Optional[Any] = None,
        supraspine: Optional[Any] = None,
        damping = 1,
        viscosity = 0,
        gravity = [0, 0, 0],
        body_mass: np.ndarray = [0, 2, 1, 0],
        muscle_strength: float = 1,
        relative_muscle_scale: np.ndarray = None,
        episode_length: int = 100,
        seed: Optional[int] = 12345,
        ic: Optional[np.ndarray] = None,
        ic_range: Optional[np.ndarray] = None,
        random_ic: bool = False,
        linked: str = 'none',
        ic_seed: Optional[int] = None,
        xml_path: str = 'arm26.xml',
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

        self.type = 'arm26babble'
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

        # Update body masses via MjSpec and recompile so MuJoCo
        # recomputes inertia/ipos exactly.
        # Must happen BEFORE spine/supraspine init (they cache model refs).
        if xml_path == 'arm26.xml':
            for geom_spec in self.spec.geoms:
                gid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_spec.name
                )
                if gid >= 0:
                    bid = self.model.geom_bodyid[gid]
                    new_mass = body_mass[bid]
                    if new_mass > 0 and geom_spec.mass > 0:
                        geom_spec.mass = float(new_mass)

            self.model, self.data = self.spec.recompile(self.model, self.data)

        # Apply non-geom mechanical properties (after recompile)
        self.model.dof_damping[:] = damping
        gravity = gravity if gravity is not None else [0, 0, 0]
        self.model.opt.gravity[:] = gravity
        gravity_bit = int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
        if np.allclose(gravity, 0):
            self.model.opt.disableflags |= gravity_bit
        else:
            self.model.opt.disableflags &= ~gravity_bit
        self.model.opt.viscosity = viscosity
        mujoco.mj_forward(self.model, self.data)

        # Now initialize spine/supraspine with the final model
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
        self._ic_seed_explicit = ic_seed is not None
        self.ic_seed = ic_seed if ic_seed is not None else seed

        np.random.seed(self.seed)
        self.ic_rng = np.random.default_rng(self.ic_seed)

        # Free-vibration initial conditions: [q0, q1, dq0, dq1]
        self.ic = np.zeros(4) if ic is None else np.asarray(ic, dtype=float)
        self.ic_range = (
            np.zeros(4) if ic_range is None else np.asarray(ic_range, dtype=float)
        )
        if isinstance(random_ic, str):
            random_ic = random_ic.strip().lower() == 'true'
        self.random_ic = bool(random_ic)
        # IC linking mode -> antiphase kick by mirroring one joint onto the other:
        #   'vel-linked' -> dq1 = -dq0 ; 'pos-linked' -> q1 = -q0 ; 'none' -> as-is.
        _valid_linked = ('none', 'vel-linked', 'pos-linked')
        linked = str(linked).strip().lower()
        if linked == 'true':
            linked = 'vel-linked'      # backward-compat with the old boolean
        elif linked == 'false':
            linked = 'none'
        if linked not in _valid_linked:
            raise ValueError(f"linked={linked!r} not in {_valid_linked}")
        self.linked = linked

        # A kinematic reset (teleport to the IC) only makes sense when an IC is
        # actually requested -- pendulum / free-vibration trials. Plain babble
        # requests no IC, so we keep qpos/qvel/act continuous across episodes and
        # let the passive muscle mechanics keep the arm bounded; otherwise every
        # episode boundary injects an artificial jump into the spontaneous activity.
        self.reset_kinematics = bool(
            self.random_ic
            or np.any(self.ic != 0)
            or np.any(self.ic_range != 0)
        )

        if relative_muscle_scale is None:
            relative_muscle_scale = np.ones(self.model.nu)
        self.relative_muscle_scale = relative_muscle_scale

        # Scale the muscle's peak force F0 (gainprm[:,2]) directly, NOT `scale`
        # (gainprm[:,3]). The XML sets an explicit force>=0, so MuJoCo ignores
        # `scale` and no longer auto-sizes F0 = scale/acc0 -- keeping F0 (hence
        # muscle strength) decoupled from limb inertia (body_mass) rather than
        # secretly co-varying with it across the embodiment sweep.
        self.default_muscle_gain = (
                relative_muscle_scale * self.model.actuator_gainprm[:,2]
        )
        self.default_muscle_bias = (
                relative_muscle_scale * self.model.actuator_biasprm[:,2]
        )
        self.muscle_strength = np.array([muscle_strength])
        self.set_muscle_strength(self.muscle_strength)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        np.set_printoptions(precision=2, suppress=True)
        print('\n--- Environment Properties ---')
        print(f'Model: {xml_path}\n')
        print('body_mass: ', self.model.body_mass)
        print('body inertia: \n', self.model.body_inertia)
        print(f'gravity: {self.model.opt.gravity}')
        print(f'viscosity: {self.model.opt.viscosity}')
        print(f'damping: {self.model.dof_damping}')
        print(f'reset kinematics per episode: {self.reset_kinematics}')
        print('Muscle Normalization:')
        # gainprm = [range0, range1, force, scale, lmin, lmax, vmax, fpmax, fvmax]
        # Everything needed to reconstruct the length map (L0, LT, range) and the
        # FLV(lmin, lmax, vmax, fpmax, fvmax) force curves for any simulated model.
        gainprm = self.model.actuator_gainprm
        print('range: \n', gainprm[:, 0:2])   # [range0, range1], operating length in L0 units
        print('force: ', gainprm[:, 2])       # peak active force F0 (-1 => derived from scale)
        print('scale: ', gainprm[:, 3])       # force scale (current strength knob)
        print('L0: ', self.spine.L0)
        print('LT: ', self.spine.LT)          # fixed tendon offset: len = LT + L0 * L_tilde
        print('lmin: ', self.spine.lmin)
        print('lmax: ', self.spine.lmax)
        print('vmax: ', self.spine.vmax)
        print('fpmax: ', gainprm[:, 7])       # peak passive force at lmax (units of F0)
        print('fvmax: ', gainprm[:, 8])       # peak force at max lengthening velocity (units of F0)
        self.spine._sample_mechanics(self.data)
        # At rest (q=0, vertical arm) we want scaled_length == 1.0: the peak of the
        # active FL curve and the toe of the passive FL curve (passive force starts here).
        #   scaled_length : fiber length in L0 units, 1.0 = optimal              (L0 units)
        #   length        : afferent signal, min-max over [lmin,lmax], 0..1      (0.5 <=> L~=1.0)
        #   force         : actuator_force with zero drive => passive force      (Newtons)
        print('scaled_length at rest [L0 units, 1.0=optimal]: ', self.spine.scaled_length)
        print('length at rest        [afferent 0..1]:         ', self.spine.length)
        print('passive force at rest [N, zero drive]:         ', self.spine.force)
        print()
        np.set_printoptions()

        # Cache for the tb log file
        self.log_eps = 0
        self.log_eps_reward = 0

    def set_muscle_strength(self, strength):
        self.muscle_strength[:] = strength
        self.model.actuator_gainprm[:,2] = (
            strength * self.default_muscle_gain
        )
        self.model.actuator_biasprm[:,2] = (
            strength * self.default_muscle_bias
        )



        #print('Muscle strength set to: ', self.model.actuator_gainprm[:,3])

    def update_seed(self, seed):
        self.seed = int(seed)
        np.random.seed(self.seed)
        # A dedicated ic_seed stays fixed across runs; otherwise IC sampling
        # follows the main seed.
        if not self._ic_seed_explicit:
            self.ic_seed = self.seed
            self.ic_rng = np.random.default_rng(self.ic_seed)
        print(f'Seed: {self.seed}\n\n')


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
        if self.reset_kinematics:
            # Explicit IC requested: clear qpos/qvel/act and impose the
            # free-vibration IC [q0, q1, dq0, dq1].
            mujoco.mj_resetData(self.model, self.data)
            if self.random_ic:
                ic = self.ic_rng.uniform(-self.ic_range, self.ic_range)
            else:
                ic = self.ic.copy()
            if self.linked == 'vel-linked':
                ic[3] = -ic[2]   # elbow vel = -(shoulder vel): antiphase velocity
            elif self.linked == 'pos-linked':
                ic[1] = -ic[0]   # elbow angle = -(shoulder angle): antiphase position
            self.data.qpos[:2] = ic[:2]
            self.data.qvel[:2] = ic[2:]
            mujoco.mj_forward(self.model, self.data)
            self.spine.set_descending_drive(np.zeros(self.spine.n,))
        # Episode bookkeeping always resets. When no IC is requested the kinematic
        # state (qpos/qvel/act) is left untouched so spontaneous babble continues
        # smoothly across episode boundaries.
        self.data.time = 0
        self.simulation_steps = 0

    def _update_status(self):
        if self.simulation_steps >= self.simulation_steps_per_episode:
            self.done = True
            self.eps += 1

    def step(self, action):
        self.reward = 0

        drive = self.supraspine.forward(self.data);
        self.spine.set_descending_drive(drive)

        for steps in range(self.supraspine.decimation_factor):
            mujoco.mj_step(self.model, self.data)
            self.simulation_steps += 1
            self._update_figures()
            self._render()
            self._update_status()
            self._write_data_to_logfile()

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


