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
