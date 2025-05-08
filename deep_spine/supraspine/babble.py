import numpy as np
from deep_spine.supraspine.base_supraspine import BaseSupraspine
from deep_spine.spine.filters import iir

class Babble(BaseSupraspine):
    """Generate random motor babbling for exploration."""

    def __init__(
        self,
        decimation_factor: int = 1,
        babble_max_prob=0.005,
        babble_duration_range=np.array([0.05, 0.1]),
        babble_rise_time_range=np.array([0.5, 0.8]),
        seed=12345
    ):
        super().__init__(
                decimation_factor=decimation_factor
        )

        # Babbling parameters
        self.babble_max_prob = babble_max_prob
        self.babble_duration_range = babble_duration_range
        self.babble_rise_time_range = babble_rise_time_range

        # Random number generator
        self.rng = np.random.default_rng(seed)

    def update_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def set_babble_max_prob(self, babble_max_prob):
        self.babble_max_prob = babble_max_prob

    def _initialize(self):
        # State variables
        self.babble_signal = np.zeros(self.n)
        self.babble_prob = np.zeros(self.n)
        self.babble_t0 = np.zeros(self.n)
        self.babble_amp = np.zeros(self.n)
        self.babble_duration = np.zeros(self.n)
        self.babble_rise_time = np.zeros(self.n)

        # Filters for smooth activation
        self.babble_filter = [
            iir.IirFilt(
                [0, self.babble_rise_time_range[0]],
                [1, self.babble_rise_time_range[0] - 1]
            ) for _ in range(self.n)
        ]

    def forward(self, data):
        """Generate new babble events and update current signals."""
        # Check for expired babble events and potentially create new ones
        for i in range(self.n):
            if (
                (data.time >= self.babble_duration[i] + self.babble_t0[i])
                or (self.babble_prob[i] == 0)
                or (data.time == 0)
            ):
                self.babble_prob[i] = (
                    self.rng.uniform(0, 1) <= self.babble_max_prob
                )
                if self.babble_prob[i]:
                    self.babble_t0[i] = data.time
                    self.babble_amp[i] = self.rng.uniform(0, 1)
                    self.babble_duration[i] = self.rng.uniform(
                        self.babble_duration_range[0],
                        self.babble_duration_range[1]
                    )
                    self.babble_rise_time[i] = self.rng.uniform(
                        self.babble_rise_time_range[0],
                        self.babble_rise_time_range[1]
                    )
                    self.babble_filter[i] = iir.IirFilt(
                        [0, self.babble_rise_time[i]],
                        [1, self.babble_rise_time[i] - 1]
                    )

        # Update signal amplitudes based on current time
        self.drive[:] = np.array([
            self.babble_prob[i]
            * self.babble_amp[i]
            * self.babble_filter[i].filter(
                data.time <= self.babble_duration[i] + self.babble_t0[i]
            ) for i in range(self.n)
        ])

        return self.drive
