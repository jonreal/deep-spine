import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

import deep_spine.spine.filters.iir as iir
import deep_spine.spine.filters.fir as fir

#class Neuron(nn.Module):
#    def __init__(
#        self,
#        num_motor_units,

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
        K=0
    ):
        super().__init__()
        self.n = num_motor_units
        self.II_gain = II_gain
        self.vel_gain = vel_gain
        self.saturation_constant = saturation_constant
        self.K = K

    def forward(self, velocity, act_II, act_muscle):
        activity_vector = (
            ((self.saturation_constant + torch.log10(act_muscle + 0.1))
            * self.vel_gain * velocity
            + self.II_gain * act_II
            + act_muscle) / 2
        )
        return torch.clamp(activity_vector, 0, 1)

class aMN(nn.Module):
    def __init__(
        self,
        num_motor_units,
        K,
        k_s=0.1,
        k_d=1.1,
        p=1,
        W_Ia=None,
        shunt_model='linear',
        seed=12345
    ):
        super().__init__()
        self.n = num_motor_units
        self.K = K
        self.k_s = torch.tensor(k_s)
        self.k_d = torch.tensor(k_d)
        self.p = torch.tensor(1)
        torch.manual_seed(seed)
        if W_Ia is None:
            self.W_Ia = nn.Parameter(
                torch.normal(mean=0.3, std=0.03,  size=(self.n, self.n)),
                requires_grad=False
            )
        else:
            self.W_Ia = nn.Parameter(
                W_Ia.clone().detach(),
                requires_grad=False
            )

        self.k_filter = (
            [iir.IirFilt([0, self.K], [1, K-1]) for _ in range(self.n)]
        )

        shunt_models = {
            'linear': self.linear_shunt,
            'log': self.log_shunt,
            'power': self.power_shunt
        }
        self.shunt_model = shunt_models[shunt_model]

        print(self.W_Ia)


    @torch.no_grad()
    def forward(self, act_Ia, descending_drive):
        synaptic_act = torch.zeros(self.n)
        for i in range(self.n):
            for j in range(self.n):
                synaptic_act[i] += (
                    act_Ia[j] * torch.clip(self.W_Ia[i, j], -1, 1)
                )
        activity = synaptic_act + descending_drive
        open_channels = torch.abs(activity)
        a = torch.tensor([
            self.k_filter[i].filter(
                activity[i] / self.shunt_model(open_channels[i])
            ) for i in range(self.n)
        ])
        return torch.clip(a, 0, 1)

    def linear_shunt(self, open_channels):
        return self.k_s + self.k_d * open_channels

    def log_shunt(self, open_channels):
        return self.k_s + self.k_d * torch.log(1 + open_channels)

    def power_shunt(self, open_channels):
        return self.k_s + self.k_d * open_channels**self.p

    def exp_shunt(self, open_channels):
        return self.k_s + self.k_d * np.exp(open_channels)

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
