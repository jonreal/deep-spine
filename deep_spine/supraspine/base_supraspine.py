from abc import ABC, abstractmethod
import numpy as np

class BaseSupraspine(ABC):
    """Base class for supraspinal control layers."""

    def __init__(
        self,
        decimation_factor: int = 1,
        ):
        self.decimation_factor = decimation_factor

    def initialize(self, model, data):
        self.n = model.na
        self.drive = np.zeros(self.n)
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        raise NotImplementedError("missing '_initialize' method")

    @abstractmethod
    def forward(self, data) -> None:
        raise NotImplementedError("missing 'forward' method")

