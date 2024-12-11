from typing import Optional, Any
import gymnasium as gym
import numpy as np
import mujoco as mj

from stable_baselines3.common.callbacks import BaseCallback
from deep_spine.enviroments.base_environment import BaseEnvironment

class Arm26(BaseEnvironment, gym.Env):
    def __init__(
            self,
            path_to_logfile: Optional[str] = None,
            simulation_sampling_rate: int = 200,
            task: Optional[str] = 'balance,
