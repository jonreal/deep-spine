import numpy as np

class IirFilt(object):
    def __init__(self, b, a):
        if a is None or b is None:
            self.identity = True
        else:
            self.identity=False
            self.b = np.asarray(b)
            self.a = np.asarray(a)
            self.n = len(b)
            self.x = np.zeros(self.n)
            self.y = np.zeros(self.n)


    def filter(self,sample):
        if not self.identity:
            self.x[1:] = self.x[:-1]
            self.x[0] = sample
            out = (
                (np.dot(self.x,self.b) - np.dot(self.y[:-1],self.a[1:]))
                /self.a[0]
            )
            self.y[1:] = self.y[:-1]
            self.y[0] = out

        else:
            out = sample
        return out


class IirBatch:
    """Vectorized IIR filter for parallel channels with identical coefficients.

    All channels share the same b, a coefficients but maintain independent state.
    Assumes order 1 (n=2) filters.
    """
    def __init__(self, b, a, shape):
        self.b = np.asarray(b, dtype=np.float64)
        self.a = np.asarray(a, dtype=np.float64)
        self.shape = shape
        self.x = np.zeros((*shape, 2))
        self.y = np.zeros((*shape, 2))

    def filter(self, samples):
        """Filter an array of samples in parallel.

        Args:
            samples: array of shape self.shape
        Returns:
            out: filtered array of shape self.shape
        """
        self.x[..., 1] = self.x[..., 0]
        self.x[..., 0] = samples
        out = (self.x[..., 0] * self.b[0] + self.x[..., 1] * self.b[1]
               - self.y[..., 0] * self.a[1]) / self.a[0]
        self.y[..., 1] = self.y[..., 0]
        self.y[..., 0] = out
        return out
