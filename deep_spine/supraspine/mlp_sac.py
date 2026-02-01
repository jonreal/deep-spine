import types
import numpy as np
from deep_spine.supraspine.base_supraspine import BaseSupraspine
from deep_spine.spine.filters import iir
import torch
from stable_baselines3 import SAC
from gymnasium import spaces


class MlpSAC(BaseSupraspine):

    def __init__(
        self,
        kwargs = None,
        decimation_factor: int = 1,
        seed= 12345
    ):

        super().__init__(
                decimation_factor=decimation_factor
        )
        self.kwargs = kwargs


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
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[16,16],qf=[16,16])
        )
        self.model = SAC("MlpPolicy", self.env,
            policy_kwargs=policy_kwargs, **self.kwargs,
        )

    def sample(self):
        return np.concatenate(
            (
                self.env.spine.sample(),
                np.array(
                    [self.env.data.qpos[-1].copy(),
                     self.env.data.qvel[-1].copy()
                    ]
                ),
            )
        )

    def forward(self, action):
        self.drive[:] = action
        return self.drive

