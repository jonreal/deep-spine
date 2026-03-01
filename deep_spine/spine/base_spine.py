from abc import ABC, abstractmethod
import os
import numpy as np
from gymnasium import spaces
import sys
import torch

def normalize(x, xmin, xmax):
    return  (x - xmin) / (xmax - xmin)

def init_weight_matrix(value, shape):
    """Initialize weights based on the form of `value`."""
    if value == 0:
        return torch.zeros(shape, dtype=torch.float32)
    elif isinstance(value, str) and os.path.isfile(value):
        data = np.loadtxt(value, delimiter=",")
        last_row = data[-1]
        flat_size = np.prod(shape)
        return torch.tensor(
            last_row[-flat_size:].reshape(shape), dtype=torch.float32
        )
    elif isinstance(value, torch.Tensor):
        return value.clone().detach()
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        mu, sigma = value
        return torch.normal(mean=mu, std=sigma, size=shape)
    else:
        # default random normal
        return torch.normal(mean=0.2, std=0.01, size=shape)

class BaseSpine:
    def __init__(
        self,
        decimation_factor = 1,
    ):
        self.decimation_factor = decimation_factor
        self.simulation_frame_count = 1
        self.learners = []

    def initialize(self, model, data) -> None:
        self.n = model.na
        lengthrange = model.actuator_lengthrange
        Range = model.actuator_gainprm[:,0:2]

        self.lmin = model.actuator_gainprm[:,4]
        self.lmax = model.actuator_gainprm[:,5]
        self.vmax = model.actuator_gainprm[:,6]

        self.L0 = (
            (lengthrange[:,1] - lengthrange[:,0])
            / (Range[:,1] - Range[:,0])
        )
        self.LT = lengthrange[:,0] - Range[:,0] * self.L0

        # allocate the raw 'sensor' signals
        self.scaled_length = np.zeros(self.n)
        self.scaled_velocity = np.zeros(self.n)
        self.length = np.zeros(self.n)
        self.velocity = np.zeros(self.n)
        self.force = np.zeros(self.n)
        self.force_signed = np.zeros(self.n)

        # allocate the spinal signals
        self.act = np.zeros(self.n) # current activation of muscles
        self.drive = np.zeros(self.n) # descending drive to muscles

        # precompute sign pattern for force_signed
        self.force_signs = np.array([(-1.0) ** (i+1) for i in range(self.n)])

        self._initialize()

    def _sample_mechanics(self, data):
        self.scaled_length[:] = (data.actuator_length[:self.n] - self.LT) / self.L0
        self.scaled_velocity[:] = data.actuator_velocity[:self.n] / self.L0
        self.length[:] = normalize(self.scaled_length, self.lmin, self.lmax)
        self.velocity[:] = self.scaled_velocity / self.vmax
        self.force[:] = data.actuator_force[:self.n]
        self.force_signed[:] = self.force_signs * self.force
        self.act[:] = data.act[:self.n]

    def set_learners(self, learners):
        self.learners = learners
        if learners:
            self.type = f'{self.type}_heb'

    def callback(self, model, data) -> None:
        if (self.simulation_frame_count == self.decimation_factor
            or data.time == 0.0
        ):
            # Activate muscles
            data.ctrl[:self.n] = self.forward(data)
            for learner in self.learners:
                learner.step()

            self.simulation_frame_count = 1
        else:
            self.simulation_frame_count += 1

    def set_descending_drive(self, drive) -> None:
        self.drive = drive

    @abstractmethod
    def _initialize(self) -> None:
        raise NotImplementedError("missing '_initialize' method")

    @abstractmethod
    def forward(self, model, data) -> None:
        raise NotImplementedError("missing 'forward' method")
