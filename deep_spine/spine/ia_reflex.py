import os
from typing import Callable, Tuple
from gymnasium import spaces
import torch
from torch import nn
import numpy as np
from functools import partial

from deep_spine.spine.base_spine import BaseSpine
from deep_spine.spine.spinal_layers import *

def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def init_weight_matrix(value, shape):
    """Initialize weights based on the form of `value`."""
    if value == 0:
        return torch.zeros(shape, dtype=torch.float32)
    elif isinstance(value, str) and os.path.isfile(value):
        data = np.loadtxt(value, delimiter=",")
        last_row = data[-1]
        flat_size = np.prod(shape)
        return torch.tensor(last_row[-flat_size:].reshape(shape), dtype=torch.float32)
    elif isinstance(value, torch.Tensor):
        return value.clone().detach()
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        mu, sigma = value
        return torch.normal(mean=mu, std=sigma, size=shape)
    else:
        # default random normal
        return torch.normal(mean=0.2, std=0.01, size=shape)

class IaReflex(nn.Module):
    def __init__(
        self,
        num_motor_units = 0,
        II_length_gain = 0,
        II_bias = 0,
        Ia_II_gain = 0,
        Ia_saturation_constant = 0,
        Ia_velocity_gain = 0,
        muscle_bias = 0,
        muscle_gain = 0,
        kalman_gain = 0,
        static_leak = 0.1,
        dynamic_leak = 1.1,
        shunt_model='linear',
        W_Ia = None,
        ):
        super().__init__()
        self.n = num_motor_units
        weights = init_weight_matrix(W_Ia, (self.n, self.n))
        self.W_Ia = nn.Parameter(weights, requires_grad=False)

        #if W_Ia is None:
        #    self.W_Ia = nn.Parameter(
        #        torch.normal(mean=0.3, std=0.03,  size=(self.n, self.n)),
        #        requires_grad=False
        #    )
        #else:
        #    self.W_Ia = nn.Parameter(
        #        W_Ia.clone().detach(),requires_grad=False
        #    )

        self.II = II(
            self.n,
            bias=II_bias,
            length_gain=II_length_gain
        )
        self.Ia = Ia(
            self.n,
            II_gain=Ia_II_gain,
            saturation_constant=Ia_saturation_constant,
            vel_gain=Ia_velocity_gain
        )
        self.MN = Neuron(
            self.n,
            kalman_gain=kalman_gain,
            static_leak=static_leak,
            dynamic_leak=dynamic_leak,
            shunt_model=shunt_model,
        )
        self.muscle = Muscle(
            self.n,
            bias=muscle_bias,
            gain=muscle_gain
        )

    @torch.no_grad()
    def forward(self, features):

        # Features: [length, velocity, activation, descending_drive]
        length = features[:self.n]
        velocity = features[self.n:self.n*2]
        activation = features[self.n*2:self.n*3]
        drive = features[self.n*3:]

        act_II = self.II(length, activation)
        act_Ia = self.Ia(velocity, act_II, activation)

        #synaptic_activation = drive
        #for i in range(self.n):
        #    for j in range(self.n):
        #        synaptic_activation[i] += (
        #            act_Ia[j] * torch.clip(self.Wp_Ia[i, j], 0, 1)
        #            -1 * act_Ia[j] * torch.clip(self.Wn_Ia[i, j], 0, 1)
        #        )

        # Clamped
        #synaptic_activation = (
        #    torch.matmul(
        #        torch.clip(self.Wp_Ia, 0, 1) - torch.clip(self.Wn_Ia, 0, 1),
        #        act_Ia
        #    ) + drive
        #)

        # Not clamped
        synaptic_activation = torch.matmul(self.W_Ia, act_Ia) + drive
        open_channels = torch.abs(torch.matmul(self.W_Ia, act_Ia) + drive)

        act_MN = self.MN(synaptic_activation,open_channels)
        act_muscle = self.muscle(act_MN)

        return act_II, act_Ia, act_MN, act_muscle

class IaReflexSpine(BaseSpine):
    def __init__(
        self,
        **iareflex_kwargs
    ):
        base_args = {
            key: iareflex_kwargs.pop(key) for key in [
                'decimation_factor', 'mechanics_filter_cutoff'
            ]
        }
        super().__init__(**base_args)
        self.type = 'ia_reflex'
        self.kwargs = iareflex_kwargs

    def _initialize(self):
        self.net = IaReflex(num_motor_units=self.n, **self.kwargs)
        self.act_II = np.zeros(self.n)
        self.act_Ia = np.zeros(self.n)
        self.act_MN = np.zeros(self.n)
        self.act_muscle = np.zeros(self.n)

    def sample(self):
        return as_numpy(self.act_Ia)

    def forward(self, data) -> np.ndarray:
        self._sample_mechanics(data)

        features = np.concatenate(
            [self.length, self.velocity, self.act, self.drive]
        )
        features = torch.tensor(features, dtype=torch.float32)

        act_II, act_Ia, act_MN, act_muscle = self.net.forward(features)
        self.act_II[:] = act_II.detach().cpu().numpy()
        self.act_Ia[:] = act_Ia.detach().cpu().numpy()
        self.act_MN[:] = act_MN.detach().cpu().numpy()
        self.act_muscle[:] = act_muscle.detach().cpu().numpy()
        return self.act_muscle

