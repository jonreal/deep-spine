import numpy as np
from collections import deque


class DelayBuffer:
    """Pure time delay using ring buffer with separate push/read.
    O(1) per operation."""

    def __init__(self, delay_samples, init_value=None):
        """
        Args:
            delay_samples: Number of samples to delay (0 = no delay)
            init_value: Value to return before buffer fills (None = pass-through)
        """
        self.delay = int(delay_samples)
        self.init_value = init_value
        self.buffer = deque(maxlen=self.delay + 1) if self.delay > 0 else None
        self._latest = init_value

    def push(self, x):
        """Add new sample to buffer. Called every sim step."""
        self._latest = np.asarray(x).copy()
        if self.buffer is not None:
            self.buffer.append(self._latest)

    def sample(self):
        """Get delayed sample."""
        if self.delay == 0:
            return self._latest
        if self.buffer is None or len(self.buffer) < self.buffer.maxlen:
            return self.init_value if self.init_value is not None else self._latest
        return self.buffer[0]

    def reset(self):
        """Clear the buffer."""
        if self.buffer is not None:
            self.buffer.clear()
        self._latest = self.init_value


class TapBuffer:
    """Tapped delay line: concatenates current obs with k past observations.

    push_and_get(obs) inserts the newest obs and returns
    [obs_t, obs_t-1, ..., obs_t-k] concatenated (newest first).
    Initialized with zeros; no warm-up period needed.
    """

    def __init__(self, n_taps, obs_dim):
        self.n_taps = n_taps
        self.obs_dim = obs_dim
        self.history = deque(
            [np.zeros(obs_dim) for _ in range(n_taps + 1)],
            maxlen=n_taps + 1
        )

    def push_and_get(self, obs):
        self.history.appendleft(np.asarray(obs).copy())
        return np.concatenate(list(self.history))

    def reset(self):
        self.history = deque(
            [np.zeros(self.obs_dim) for _ in range(self.n_taps + 1)],
            maxlen=self.n_taps + 1
        )
