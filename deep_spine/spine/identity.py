import numpy as np
from gymnasium import spaces
import torch
from torch import nn

from deep_spine.spine.base_spine import BaseSpine
from deep_spine.spine.spinal_layers import *

def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

class Identity(nn.Module):
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
        ):
        super().__init__()
        self.n = num_motor_units

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

        # Features: [length, velocity, activation]
        length = features[:self.n]
        velocity = features[self.n:self.n*2]
        activation = features[self.n*2:self.n*3]
        drive = features[self.n*3:]

        act_II = self.II(length, activation)
        act_Ia = self.Ia(velocity, act_II, activation)


        # Not clamped
        synaptic_activation = drive
        open_channels = drive

        act_MN = self.MN(synaptic_activation,open_channels)
        act_muscle = self.muscle(act_MN)

        return act_II, act_Ia, act_MN, act_muscle

class IdentitySpine(BaseSpine):
    def __init__(
        self,
        **identity_kwargs
    ):
        base_args = {
            key: identity_kwargs.pop(key) for key in [
                'decimation_factor', 'mechanics_filter_cutoff'
            ]
        }
        super().__init__(**base_args)
        self.type = 'identity'
        self.kwargs = identity_kwargs

    def _initialize(self):
        self.net = Identity(num_motor_units=self.n, **self.kwargs)
        self.act_Ia = np.zeros(self.n);
        self.act_II = np.zeros(self.n);
        self.act_MN = np.zeros(self.n);
        self.act_muscle = np.zeros(self.n);

    def sample(self):
        return as_numpy(self.act_Ia)

    @torch.no_grad()
    def forward(self, data) -> np.ndarray:
        self._sample_mechanics(data)

        features = np.concatenate(
            [self.length, self.velocity, self.act, self.drive]
        )
        features = torch.as_tensor(features, dtype=torch.float32)

        act_II, act_Ia, act_MN, act_muscle = self.net.forward(features)
        self.act_II[:] = act_II.detach().cpu().numpy()
        self.act_Ia[:] = act_Ia.detach().cpu().numpy()
        self.act_MN[:] = act_MN.detach().cpu().numpy()
        self.act_muscle[:] = act_muscle.detach().cpu().numpy()

        return self.act_muscle
