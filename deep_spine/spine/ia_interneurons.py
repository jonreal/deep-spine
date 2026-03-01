from typing import Callable, Tuple
from gymnasium import spaces
import torch
from torch import nn
import numpy as np
from functools import partial

from deep_spine.spine.base_spine import BaseSpine, init_weight_matrix
from deep_spine.spine.spinal_layers import *

class IaInterneurons(nn.Module):
    def __init__(
        self,
        num_motor_units = 0,
        num_of_interneurons = 0,
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
        W_MN_Ia = None,
        W_MN_i = None,
        W_MN_e = None,
        W_e_e = None,
        W_e_i = None,
        W_e_Ia = None,
        W_i_i = None,
        W_i_e = None,
        W_i_Ia = None
        ):
        super().__init__()
        self.n = num_motor_units
        self.n_IN = num_of_interneurons

        weights = [
            ("W_MN_Ia", (self.n, self.n)),
            ("W_MN_i",  (self.n, self.n_IN)),
            ("W_MN_e",  (self.n, self.n_IN)),
            ("W_e_e",   (self.n_IN, self.n_IN)),
            ("W_e_i",   (self.n_IN, self.n_IN)),
            ("W_e_Ia",  (self.n_IN, self.n)),
            ("W_i_i",   (self.n_IN, self.n_IN)),
            ("W_i_e",   (self.n_IN, self.n_IN)),
            ("W_i_Ia",  (self.n_IN, self.n)),
        ]

        for name, shape in weights:
            val = init_weight_matrix(locals()[name], shape)
            setattr(self, name, nn.Parameter(val, requires_grad=False))


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

        self.INe =  Neuron(
            self.n_IN,
            kalman_gain=kalman_gain,
            static_leak=static_leak,
            dynamic_leak=dynamic_leak,
            shunt_model=shunt_model,
        )
        self.INi =  Neuron(
            self.n_IN,
            kalman_gain=kalman_gain,
            static_leak=static_leak,
            dynamic_leak=dynamic_leak,
            shunt_model=shunt_model,
        )

    @torch.no_grad()
    def forward(self, features):

        # Features:
        #   [
        #       length, velocity, activation, descending_drive
        #       act_INe, act_INa
        #   ]
        length = features[:self.n]
        velocity = features[self.n:self.n*2]
        activation = features[self.n*2:self.n*3]
        drive = features[self.n*3:self.n*4]

        act_INe = features[self.n*4:self.n*4 + self.n_IN]
        act_INi = features[self.n*4 + self.n_IN:]

        act_II = self.II(length, activation)
        act_Ia = self.Ia(velocity, act_II, activation)

        # Excitatory interneuron synaptic currents (compute once)
        ee_current  = torch.matmul(self.W_e_e, act_INe)
        ei_current  = torch.matmul(self.W_e_i, act_INi)
        eIa_current = torch.matmul(self.W_e_Ia, act_Ia)

        # Net current (signed) vs total conductance (unsigned)
        synaptic_activation_INe = ee_current - ei_current + eIa_current
        open_channels_INe       = ee_current + ei_current + eIa_current

        # Inhibitory interneuron synaptic currents (compute once)
        ii_current  = torch.matmul(self.W_i_i, act_INi)
        ie_current  = torch.matmul(self.W_i_e, act_INe)
        iIa_current = torch.matmul(self.W_i_Ia, act_Ia)

        # Net current (signed) vs total conductance (unsigned)
        synaptic_activation_INi = -ii_current + ie_current + iIa_current
        open_channels_INi       =  ii_current + ie_current + iIa_current

        act_INe = self.INe(synaptic_activation_INe, open_channels_INe)
        act_INi = self.INi(synaptic_activation_INi, open_channels_INi)

        act_INe = act_INe.to(dtype=self.W_MN_e.dtype)
        act_INi = act_INi.to(dtype=self.W_MN_i.dtype)

        # Motor neuron synaptic currents (compute once)
        mn_Ia_current = torch.matmul(self.W_MN_Ia, act_Ia)
        mn_e_current  = torch.matmul(self.W_MN_e, act_INe)
        mn_i_current  = torch.matmul(self.W_MN_i, act_INi)

        # Net current (signed) vs total conductance (unsigned)
        synaptic_activation_MN = mn_Ia_current + mn_e_current - mn_i_current + drive
        open_channels_MN       = mn_Ia_current + mn_e_current + mn_i_current + drive

        act_MN = self.MN(synaptic_activation_MN, open_channels_MN)
        act_muscle = self.muscle(act_MN)

        return act_II, act_Ia, act_MN, act_muscle, act_INe, act_INi

class IaInterneuronSpine(BaseSpine):
    def __init__(
        self,
        **ia_interneurons_kwargs
    ):
        base_args = {
            key: ia_interneurons_kwargs.pop(key) for key in [
                'decimation_factor',
            ]
        }
        ia_interneurons_kwargs.pop('mechanics_filter_cutoff', None)
        super().__init__(**base_args)
        self.type = 'ia_interneurons'
        self.kwargs = ia_interneurons_kwargs

    def _initialize(self):
        self.net = IaInterneurons(
            num_motor_units=self.n,
            **self.kwargs
        )
        self.act_II = np.zeros(self.n)
        self.act_Ia = np.zeros(self.n)
        self.act_MN = np.zeros(self.n)
        self.act_muscle = np.zeros(self.n)

        self.act_INe = np.zeros(self.net.n_IN)
        self.act_INi = np.zeros(self.net.n_IN)

    def forward(self, data) -> np.ndarray:
        self._sample_mechanics(data)

        features = np.concatenate([
            self.length, self.velocity, self.act, self.drive,
            self.act_INe, self.act_INi
        ])
        features = torch.tensor(features, dtype=torch.float32)

        act_II, act_Ia, act_MN, act_muscle, act_INe, act_INi = (
            self.net.forward(features)
        )

        self.act_II[:] = act_II.detach().cpu().numpy()
        self.act_Ia[:] = act_Ia.detach().cpu().numpy()
        self.act_MN[:] = act_MN.detach().cpu().numpy()
        self.act_muscle[:] = act_muscle.detach().cpu().numpy()

        self.act_INe = act_INe.detach().cpu().numpy()
        self.act_INi = act_INi.detach().cpu().numpy()

        return self.act_muscle

