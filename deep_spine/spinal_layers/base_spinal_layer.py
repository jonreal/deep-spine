from abc import ABC, abstractmethod
import numpy as np
from scipy import signal
from gymnasium import spaces
import sys

import deep_spine.spinal_layers.filters.iir as iir

def normalize_length(length, min_length, max_length):
    return max(
        0,
        min(1, (length - min_length) / (max_length - min_length))
    )
def normalize_velocity(velocity, min_length, max_length):
    return velocity / (max_length - min_length)

class BaseSpinalLayer:
    def __init__(
        self,
        decimation_factor = 1,
        mechanics_filter_cutoff = None
    ):
        self.decimation_factor = decimation_factor
        self.mechanics_filter_cutoff = mechanics_filter_cutoff
        self.simulation_frame_count = 1

    def initialize(self, model) -> None:
        self.n = model.na
        lengthrange = model.actuator_lengthrange
        self.min_length = lengthrange[0:self.n,0]
        self.max_length = lengthrange[0:self.n,1]

        # the raw 'sensor' signals
        self.length = np.zeros(self.n)
        self.velocity = np.zeros(self.n)
        self.force = np.zeros(self.n)

        self.activation = np.zeros(self.n) # current activation of muscles
        self.drive = np.zeros(self.n) # descending drive to muscles

        b, a = (
            signal.butter(1, self.mechanics_filter_cutoff, 'low')
            if self.mechanics_filter_cutoff else (None, None)
        )
        self.mechanics_filters = [iir.IirFilt(b, a) for _ in range(3*self.n)]

        self.initialize_layers()

    def _sample_mechanics(self, data):
        for i in range(self.n):
            self.length[i] = self.mechanics_filters[i].filter(
                normalize_length(
                    data.actuator_length[i],
                    self.min_length[i],
                    self.max_length[i]
                )
            )
            self.velocity[i] = self.mechanics_filters[i+self.n].filter(
                normalize_velocity(
                    data.actuator_velocity[i],
                    self.min_length[i],
                    self.max_length[i]
                )
            )
            self.force[i] = self.mechanics_filters[i+2*self.n].filter(
                data.actuator_force[i]
            )
            self.activation[i] = data.act[i]

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
    def initialize_layers(self) -> None:
        raise NotImplementedError("missing 'initialize_layers' method")

    @abstractmethod
    def forward(self, model, data) -> None:
        raise NotImplementedError("missing 'forward' method")
