import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

import deep_spine.spine.filters.iir as iir
import deep_spine.spine.filters.fir as fir

class II(nn.Module):
    def __init__(
        self,
        num_motor_units,
        length_gain=0,
        bias=0
        ):
        super().__init__()
        self.n = num_motor_units
        self.length_gain = length_gain
        self.bias = bias

    def forward(self, length, act_muscle):
        return torch.clamp(
            (self.length_gain * (length + self.bias) + act_muscle) / 2,
            0, 1
        )

class Ia(nn.Module):
    def __init__(
        self,
        num_motor_units,
        II_gain=0,
        vel_gain=0,
        saturation_constant=0,
    ):
        super().__init__()
        self.n = num_motor_units
        self.II_gain = II_gain
        self.vel_gain = vel_gain
        self.saturation_constant = saturation_constant

    def forward(self, velocity, act_II, act_muscle):
        activity_vector = (
            ((self.saturation_constant + torch.log10(act_muscle + 0.1))
            * self.vel_gain * velocity
            + self.II_gain * act_II
            + act_muscle) / 2
        )
        return torch.clamp(activity_vector, 0, 1)

class Neuron(nn.Module):
    def __init__(
        self,
        n,
        kalman_gain=0,
        static_leak=0.1,
        dynamic_leak=1,
        shunt_model='linear'
    ):
        super().__init__()
        self.n = n
        self.shunt = self.set_shunt_model(
            static_leak, dynamic_leak, shunt_model
        )
        self.synaptic_dynamics = iir.IirBatch(
            [0, kalman_gain], [1, kalman_gain-1], (self.n,)
        )

    def set_shunt_model(self, static_leak, dynamic_leak, shunt_model='linear'):
        if shunt_model == 'linear':
            return (
                lambda open_channels:
                static_leak + dynamic_leak * open_channels
            )
        elif shunt_model == 'log':
            return (
                lambda open_channels:
                static_leak + dynamic_leak * torch.log1p(open_channels)
            )
        else:
            raise ValueError(f"Unknown shunt model: {shunt_model}")

    @torch.no_grad()
    def forward(self, synaptic_activation, open_channels):
        potential = synaptic_activation / self.shunt(open_channels)
        a = self.synaptic_dynamics.filter(potential.numpy())
        return torch.clip(torch.from_numpy(a).float(), 0, 1)


class Muscle(nn.Module):
    def __init__(
        self,
        num_motor_units,
        bias,
        gain,
    ):
        super().__init__()
        self.n = num_motor_units
        self.bias = bias
        self.gain = gain

    def forward(self, act_aMN):
        return torch.clamp(self.gain * (act_aMN + self.bias), 0, 1)
