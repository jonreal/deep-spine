import numpy as np
from gymnasium import spaces
from deep_spine.spine.base_spine import BaseSpine


class IdentitySpine(BaseSpine):
    def __init__(
        self,
        decimation_factor: int = 1,
        mechanics_filter_cutoff: float = None
    ) -> None:
        super().__init__(
            decimation_factor=decimation_factor,
            mechanics_filter_cutoff=mechanics_filter_cutoff
        );
        self.type = 'identity'

    def forward(self, data) -> np.ndarray:
        self._sample_mechanics(data)

        # Identity - suprapinal effenent is motor unit activation
        self.act_muscle = self.drive

        return self.act_muscle
