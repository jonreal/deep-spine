import types
import numpy as np
from deep_spine.supraspine.base_supraspine import BaseSupraspine
from deep_spine.spine.filters import iir
import torch
from stable_baselines3 import PPO
from gymnasium import spaces


class MlpPPO(BaseSupraspine):

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
        self.rng = np.random.default_rng(seed)

    def _initialize(self):
        self.env.action_space = spaces.Box(
            low=np.zeros(self.n, dtype=np.float32),
            high=np.ones(self.n, dtype=np.float32),
            dtype=np.float32
        )
        self.env.observation_space = spaces.Box(
            low=np.zeros(self.n, dtype=np.float32),
            high=np.ones(self.n, dtype=np.float32),
            dtype=np.float32
        )
        policy_kwargs = dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[16,16],vf=[16,16])
        )
        self.model = PPO("MlpPolicy", self.env,
            policy_kwargs=policy_kwargs, **self.kwargs,
        )

    def forward(self, data):
        self.drive[:] = np.zeros(self.n)
        return self.drive

