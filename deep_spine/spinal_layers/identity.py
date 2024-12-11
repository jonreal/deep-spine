import numpy as np
from gymnasium import spaces
from deep_spine.spinal_layers.base_spinal_layer import BaseSpinalLayer


class IdentitySpinalLayer(BaseSpinalLayer):
    def __init__(
        self,
        decimation_factor: int = 1,
        mechanics_filter_cutoff: float = None
    ) -> None:
        super().__init__(
            decimation_factor=decimation_factor,
            mechanics_filter_cutoff=mechanics_filter_cutoff
        );
        self.type = 'identity_spinal_layer'

    def forward(self, data) -> np.ndarray:
        self._sample_mechanics(data)

        # Identity - suprapinal effenent is motor unit activation
        self.muscle_activations = self.efferent_signals

        return self.muscle_activations

    def set_efferent(self, efferent) -> None:
        self.efferent_signals = efferent
