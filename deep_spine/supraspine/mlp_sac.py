import types
import numpy as np
from deep_spine.supraspine.base_supraspine import BaseSupraspine
from deep_spine.supraspine.delay import DelayBuffer
import torch
from stable_baselines3 import SAC
from gymnasium import spaces


class MlpSAC(BaseSupraspine):

    def __init__(
        self,
        kwargs = None,
        decimation_factor: int = 1,
        proprioception_delay_steps = 0,
        visual_delay_steps = 0,
        action_delay_steps = 0,
        perception_mode = 'proprio-visual',
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

        self.perception_mode = perception_mode
        self.proprio_delay = proprioception_delay_steps;
        self.visual_delay = visual_delay_steps;
        self.action_delay = action_delay_steps;

    def update_perception(self):
        if self.perception_mode == 'proprio-visual':
            self.proprio_buffer.push(self.env.spine.sample())
        self.visual_buffer.push(np.array(
            [self.env.data.qpos[-1].copy(),
             self.env.data.qvel[-1].copy()
            ])
        )

    def update_action(self, action):
        self.action_buffer.push(action)

    def sample_action(self):
        return self.action_buffer.sample()

    def sample(self):
        if self.perception_mode == 'visual-only':
            return self.visual_buffer.sample()
        return np.concatenate(
            (
                self.proprio_buffer.sample(),
                self.visual_buffer.sample()
            )
        )

    def update_seed(self, seed):
        self.rng = np.random.default_rng(int(seed))

    def _initialize(self):

        self.drive[:] = np.zeros(self.n)

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

        self.visual_buffer = DelayBuffer(
            self.visual_delay, init_value=np.zeros(2)
        );
        self.action_buffer = DelayBuffer(
            self.action_delay, init_value=np.zeros(self.n)
        );

        # Extract net_arch from kwargs if provided, else default
        net_arch = self.kwargs.pop("net_arch", [16, 16])
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=net_arch, qf=net_arch)
        )
        self.model = SAC("MlpPolicy", self.env,
            policy_kwargs=policy_kwargs, **self.kwargs,
        )

        print('--- SAC Controller ---')
        print(f'  perception:  {self.perception_mode}')
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


    def forward(self, action):
        self.drive[:] = action
        return self.drive
