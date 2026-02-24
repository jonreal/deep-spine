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

        self.proprio_delay = proprioception_delay_steps;
        self.visual_delay = visual_delay_steps;
        self.action_delay = action_delay_steps;

    def update_perception(self):
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
        self.env.observation_space = spaces.Box(
            low=np.concatenate((
                np.zeros(self.n, dtype=np.float32),
                np.array([-np.pi,-5*np.pi], dtype=np.float32)
            )),
            high=np.concatenate((
                np.ones(self.n, dtype=np.float32),
                np.array([np.pi,5*np.pi], dtype=np.float32)
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

        # 1) Show the exact hyperparams SB3 is using (after defaults/overrides)
        print("SAC hyperparams:", self.model.get_parameters() is not None)  # sanity
        print("policy_kwargs:", self.model.policy_kwargs)
        print("lr:", self.model.learning_rate,
              "batch:", self.model.batch_size,
              "train_freq:", self.model.train_freq,
              "gradient_steps:", self.model.gradient_steps,
              "gamma:", self.model.gamma,
              "tau:", self.model.tau,
              "ent_coef:", self.model.ent_coef)


    def forward(self, action):
        self.drive[:] = action
        return self.drive
