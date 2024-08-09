from abc import ABC, abstractmethod
import numpy as np
from scipy import signal
from gymnasium import spaces

import deep_spine.spinal_layers.filters.iir as iir

class BaseSpinalLayer:
    def __init__(
        self,
        afferent_mode = 'len-vel',
        decimation_factor = 1,
        afferent_filter_cutoff = None
    ):
        self.decimation_factor = decimation_factor
        self.simulation_frame_count = 1
        self.afferent_mode = afferent_mode
        self.afferent_filter_cutoff = afferent_filter_cutoff

        self.n = None
        self.min_length = None
        self.max_length = None
        self.afferent_signals = None
        self.afferent_filters = None
        self.integrate_afferent_signals = None
        self.afferent_space = None
        self.efferent_signals = None
        self.efferent_space = None
        self.activations = None

    def initialize_motor_units(self, num_motor_units, lengthrange) -> None:
        self.n = num_motor_units
        self.min_length = lengthrange[0:self.n,0]
        self.max_length = lengthrange[0:self.n,1]

        self.afferent_signals = self._initialize_afferent_signals()
        self.afferent_filters = self._initialize_afferent_filters()
        self.integrate_afferent_signals = self._initialize_afferent_integrator()
        self.afferent_space = self._initialize_afferent_space()

        self.efferent_signals = self._initialize_efferent_signals()
        self.efferent_space = self._initialize_efferent_space()

    def _initialize_afferent_signals(self) -> np.ndarray:
        if self.afferent_mode in ('len-vel'):
            return np.zeros(2*self.n)

    def _initialize_afferent_filters(self):
        num_filters = (
            2*self.n if self.afferent_mode in ('len-vel')
                else self.n
        )
        cutoff = self.afferent_filter_cutoff
        b, a = signal.butter(1, cutoff, 'low') if cutoff else (None, None)
        return [iir.IirFilt(b, a) for _ in range(num_filters)]

    def _initialize_afferent_integrator(self):
        integrator = {
            'len-vel': self._integrate_length_velocity_afferents,
        }
        try:
            return integrator[self.afferent_mode]
        except KeyError:
            raise ValueError(f"Invalid afferent mode: {afferent_mode}.")

    def _initialize_afferent_space(self):
        if self.afferent_mode == 'len-vel':
            return spaces.Box(
                low=np.concatenate(
                    (np.zeros(self.n,),
                    -5*np.ones(self.n))
                ),
                high=np.concatenate(
                    (np.ones(self.n,),
                     5*np.ones(self.n))
                ),
                dtype=np.float32
            )


    def _integrate_length_velocity_afferents(self, data):
        self.afferent_signals[:self.n] = [
            f.filter(length / max_length) for f, length, max_length in (
                zip(
                    self.afferent_filters[:self.n],
                    data.actuator_length,
                    self.max_length
                )
            )
        ]
        self.afferent_signals[self.n:] = [
            f.filter(velocity / max_length) for f, velocity, max_length in (
                zip(
                    self.afferent_filters[self.n:],
                    data.actuator_velocity,
                    self.max_length
                )
            )
        ]

    def callback(self, model, data) -> None:
        if (self.simulation_frame_count == self.decimation_factor
            or data.time == 0.0
        ):
            # Activate muscles
            data.ctrl[:self.n] =  self._synaptic_integration(data)
            self.simulation_frame_count = 1
        else:
            self.simulation_frame_count += 1

    @abstractmethod
    def _synaptic_integration(self, model, data) -> None:
        """ Dyanamics of the spinal network """

    @abstractmethod
    def set_efferent_signals(self, new_efferent_signals) -> None:
        """ set_action """

    @abstractmethod
    def _initialize_efferent_signals(self) -> np.ndarray:
        """ Initialize efferent signals """

    @abstractmethod
    def _initialize_efferent_space(self) -> spaces.Box:
        """ Initialize efferent space """


