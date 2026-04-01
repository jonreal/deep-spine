import os
import types
import numpy as np
from scipy import signal
from deep_spine.supraspine.base_supraspine import BaseSupraspine
from deep_spine.supraspine.delay import DelayBuffer
import deep_spine.spine.filters.iir as iir
import torch
from stable_baselines3 import SAC
from gymnasium import spaces


def _make_filter(spec, shape):
    """Build an IirBatch from a filter spec dict, or return None.

    spec: {'order': 1, 'cutoff': 0.1, 'type': 'low'} or None/0/{}
    """
    if not spec:
        return None
    b, a = signal.butter(spec['order'], spec['cutoff'], spec['type'])
    return iir.IirBatch(b, a, shape)


class MlpSAC(BaseSupraspine):

    def __init__(
        self,
        kwargs = None,
        checkpoint = None,
        decimation_factor: int = 1,
        proprio_delay_steps = 0,
        visual_delay_steps = 0,
        action_delay_steps = 0,
        proprio_filter = None,
        visual_filter = None,
        action_filter = None,
        perception_mode = 'proprio-visual',
        visual_mode = 'joint',
        seed= 12345
    ):

        super().__init__(
                decimation_factor=decimation_factor
        )
        # Convert train_freq list to tuple (SB3 requires tuple)
        if kwargs and "train_freq" in kwargs and isinstance(
                kwargs["train_freq"], list):
            kwargs["train_freq"] = tuple(kwargs["train_freq"])
        self.kwargs = kwargs
        self.checkpoint = checkpoint

        self.perception_mode = perception_mode
        self.visual_mode = visual_mode
        self.proprio_delay = proprio_delay_steps;
        self.visual_delay = visual_delay_steps;
        self.action_delay = action_delay_steps;
        self.proprio_filter_spec = proprio_filter;
        self.visual_filter_spec = visual_filter;
        self.action_filter_spec = action_filter;





    def _update_perception(self):
        if self.perception_mode == 'proprio-visual':
            raw = self.env.spine.sample()
            if self.proprio_filter is not None:
                raw = self.proprio_filter.filter(raw)
            self.proprio_buffer.push(raw)
            self.proprio[:] = self.proprio_buffer.sample()

        if self.visual_mode == 'world':
            visual = np.array(
                [self.env.pendulum_pos,
                 self.env.pendulum_vel
                ])
        else:
            visual = np.array(
                [self.env.data.qpos[-1].copy(),
                 self.env.data.qvel[-1].copy()
                ])

        if self.visual_filter is not None:
            visual = self.visual_filter.filter(visual)
        self.visual_buffer.push(visual)
        self.visual[:] = self.visual_buffer.sample()

    def _update_action(self):
        if self.action_filter is not None:
            self.action[:] = self.action_filter.filter(
                np.asarray(self.action)
            )
        self.action_buffer.push(self.action)

    def update(self):
        self._update_perception();
        self._update_action();

    def sample_action(self):
        self.action[:] = self.action_buffer.sample()
        return self.action

    def sample(self):
        if self.perception_mode == 'visual-only':
            return self.visual.copy()
        return np.concatenate((self.proprio, self.visual))

    def update_seed(self, seed):
        self.rng = np.random.default_rng(int(seed))

    def _initialize(self):

        self.action = np.zeros(self.n)
        self.proprio = np.zeros(self.n)
        self.visual = np.zeros(2)

        self.env.action_space = spaces.Box(
            low=np.zeros(self.n, dtype=np.float32),
            high=np.ones(self.n, dtype=np.float32),
            dtype=np.float32
        )
        visual_low = np.array([-np.pi, -5*np.pi], dtype=np.float32)
        visual_high = np.array([np.pi, 5*np.pi], dtype=np.float32)

        if self.perception_mode == 'visual-only':
            self.env.observation_space = spaces.Box(
                low=visual_low, high=visual_high, dtype=np.float32
            )
        else:
            self.env.observation_space = spaces.Box(
                low=np.concatenate((
                    np.zeros(self.n, dtype=np.float32), visual_low
                )),
                high=np.concatenate((
                    np.ones(self.n, dtype=np.float32), visual_high
                )),
                dtype=np.float32
            )
            self.proprio_buffer = DelayBuffer(
                self.proprio_delay, init_value=np.zeros(self.n)
            );
            self.proprio_filter = _make_filter(
                self.proprio_filter_spec, (self.n,))

        self.visual_buffer = DelayBuffer(
            self.visual_delay, init_value=np.zeros(2)
        );
        self.visual_filter = _make_filter(self.visual_filter_spec, (2,))

        self.action_buffer = DelayBuffer(
            self.action_delay, init_value=np.zeros(self.n)
        );
        self.action_filter = _make_filter(
            self.action_filter_spec, (self.n,))

        if self.checkpoint:
            path = self._resolve_checkpoint(self.checkpoint)
            # Pass kwargs minus net_arch (architecture is in the checkpoint)
            load_kwargs = {k: v for k, v in self.kwargs.items()
                          if k != "net_arch"}
            self.model = SAC.load(path, env=self.env, **load_kwargs)
            print(f'  Loaded checkpoint: {path}')
        else:
            net_arch = self.kwargs.get("net_arch", [16, 16])
            model_kwargs = {k: v for k, v in self.kwargs.items()
                           if k != "net_arch"}
            policy_kwargs = dict(
                activation_fn=torch.nn.ReLU,
                net_arch=dict(pi=net_arch, qf=net_arch)
            )
            self.model = SAC("MlpPolicy", self.env,
                policy_kwargs=policy_kwargs, **model_kwargs,
            )

        print('--- SAC Controller ---')
        print(f'  perception:  {self.perception_mode}')
        print(f'  visual_mode: {self.visual_mode}')
        print(f'  obs_dim:     {self.env.observation_space.shape[0]}')
        print(f'  act_dim:     {self.env.action_space.shape[0]}')
        print(f'  net_arch:    {self.model.policy_kwargs.get("net_arch", "default")}')
        print(f'  lr:          {self.model.learning_rate}')
        print(f'  grad_steps:  {self.model.gradient_steps}')
        print(f'  train_freq:  {self.model.train_freq}')
        print(f'  batch:       {self.model.batch_size}')
        print(f'  gamma:       {self.model.gamma}')
        print(f'  tau:         {self.model.tau}')
        print(f'  ent_coef:    {self.model.ent_coef}')


    @staticmethod
    def _resolve_checkpoint(checkpoint):
        """Resolve checkpoint: explicit path to a .zip model file."""
        if isinstance(checkpoint, str) and os.path.exists(checkpoint):
            return checkpoint
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def set_action(self, action):
        self.action[:] = action
