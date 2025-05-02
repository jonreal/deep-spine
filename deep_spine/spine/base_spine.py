from abc import ABC, abstractmethod
import numpy as np
from scipy import signal
from gymnasium import spaces
import sys

import deep_spine.spine.filters.iir as iir

def normalize(x, xmin, xmax):
    return  (x - xmin) / (xmax - xmin)

class BaseSpine:
    def __init__(
        self,
        decimation_factor = 1,
        mechanics_filter_cutoff = None
    ):
        self.decimation_factor = decimation_factor
        self.mechanics_filter_cutoff = mechanics_filter_cutoff
        self.simulation_frame_count = 1

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

        b, a = (
            signal.butter(1, self.mechanics_filter_cutoff, 'low')
            if self.mechanics_filter_cutoff else (None, None)
        )
        self.mechanics_filters = [iir.IirFilt(b, a) for _ in range(3*self.n)]

        self._initialize()

    def _sample_mechanics(self, data):
        for i in range(self.n):
            self.scaled_length[i] = self.mechanics_filters[i].filter(
                (data.actuator_length[i] - self.LT[i]) / self.L0[i]
            )
            self.scaled_velocity[i] = self.mechanics_filters[i+self.n].filter(
                data.actuator_velocity[i] / self.L0[i]
            )
            self.length[i] = self.mechanics_filters[i].filter(
                normalize(
                    self.scaled_length[i],
                    self.lmin[i],
                    self.lmax[i]
                )
            )
            self.velocity[i] = self.mechanics_filters[i+self.n].filter(
                    self.scaled_velocity[i] / self.vmax[i]
            )
            self.force[i] = self.mechanics_filters[i+2*self.n].filter(
                data.actuator_force[i]
            )
            self.force_signed[i] = ((-1.0) ** (i+1)) * self.force[i]
            self.act[i] = data.act[i]

    def callback(self, model, data) -> None:
        if (self.simulation_frame_count == self.decimation_factor
            or data.time == 0.0
        ):
            # Activate muscles
            data.ctrl[:self.n] = self.forward(data)
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
