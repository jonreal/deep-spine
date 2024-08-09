import numpy as np
from gymnasium import spaces
from deep_spine.spinal_layers.base_spinal_layer import BaseSpinalLayer


class IdentitySpinalLayer(BaseSpinalLayer):
    def __init__(
        self,
        afferent_mode: str = 'len-vel',
        decimation_factor: int = 1,
        afferent_filter_cutoff: float = None
    ) -> None:
        """
        Initialize the IdentitySpinalLayer.

        Args:
            num_motor_units: Number of muscle-tendons in Mujoco model
            afferent_mode: Mode of proprioception ('len-vel', 'len', or 'vel').
            decimation_factor: Sampling rate of the spine:
            simulation_sampling_rate * decimation_factor.
            afferent_filter_cutoff: Cutoff frequency for afferent filter,
                if any.
        """
        super().__init__(
            afferent_mode=afferent_mode,
            decimation_factor=decimation_factor,
            afferent_filter_cutoff=afferent_filter_cutoff
        );
        self.type = 'identity_spinal_layer'

    def _initialize_efferent_signals(self) -> np.ndarray:
        return np.zeros(self.n)

    def _initialize_activation(self) -> np.ndarray:
        return np.zeros(self.n)

    def _initialize_efferent_space(self) -> spaces.Box:
        return spaces.Box(
            low=0, high=1.0, shape=(self.n,),dtype=np.float32
        )

    def _synaptic_integration(self, data) -> np.ndarray:
        self.integrate_afferent_signals(data)

        # Identity - suprapinal effenent is motor unit activation
        self.activations = self.efferent_signals

        return self.activations

    def set_efferent_signals(self, new_efferent_signals) -> None:
        self.efferent_signals = new_efferent_signals
