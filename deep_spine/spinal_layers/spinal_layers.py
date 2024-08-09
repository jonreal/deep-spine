import torch
from torch import nn
import torch.nn.functional as F

def create_agonist_antgonist_weight_matrix(a):
    n = len(a)
    matrix = torch.zeros((n, n))
    even_indices = torch.arange(0, len(a)-1, 2)
    odd_indices = torch.arange(1, len(a), 2)
    matrix[even_indices+1, even_indices] = a[even_indices]
    matrix[odd_indices-1, odd_indices] = a[odd_indices]
    return matrix

class IIa(nn.Module):
    def __init__(
        self,
        num_motor_units,
        length_gain=0,
        bias=0
        ):
        super().__init__()
        self.n = num_motor_units
        self.length_gain = length_gain
        self.bias = bias

    def forward(self, length, activation_MN):
        return torch.sigmoid(
            (self.length_gain * length + self.bias + activation_MN) / 2
        )

class Ia(nn.Module):
    def __init__(
        self,
        num_motor_units,
        IIa_gain=0,
        saturation_constant=0,
    ):
        super().__init__()
        self.n = num_motor_units
        self.IIa_gain = IIa_gain
        self.saturation_constant = saturation_constant

    def forward(self, velocity, activation_IIa, activation_mn):
        return torch.sigmoid(
            ((self.saturation_constant + torch.log10(activation_mn + 0.1))
            * velocity
            + self.IIa_gain * activation_IIa
            + activation_mn) / 2
        )

class aMN(nn.Module):
    def __init__(
        self,
        num_motor_units,
        bias
    ):
        super().__init__()
        self.n = num_motor_units
        self.bias = bias
        self.W_Ia = nn.Parameter(torch.randn(self.n, self.n) * 0.1 - 1.0)

    def forward(self, activation_Ia, activation_d):
        W_Ia = torch.sigmoid(self.W_Ia)
        return torch.sigmoid(
            torch.matmul(activation_Ia, W_Ia) + self.bias - activation_d
        )

class Spindle(nn.Module):
    def __init__(self, num_proprioceptive_features, num_motor_units):
        super().__init__()
        self.n = num_motor_units
        self.spindle = nn.Sequential(
            nn.Linear(num_proprioceptive_features, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, num_motor_units),
            nn.Tanh(),
        )

    def forward(self, proprioceptive_features):
        return self.spindle(proprioceptive_features)

class SpindleAndDecoder(nn.Module):
    def __init__(self, num_proprioceptive_features, num_motor_units):
        super().__init__()
        self.n = num_motor_units
        self.spindleDecoder = nn.Sequential(
            nn.Linear(num_proprioceptive_features, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2*num_motor_units),
            nn.Tanh(),
        )

    def forward(self, proprioceptive_features):
        return self.spindleDecoder(proprioceptive_features)

#        self.epsilon = torch.nn.Parameter(torch.randn(self.n))
#
#    def forward(self, length, velocity):
#        return length + torch.sigmoid(self.epsilon) * velocity

class RecInh(nn.Module):
    def __init__(self,
        num_motor_units,
        network_structure='aa'
    ):
        super().__init__()
        self.n = num_motor_units
        if network_structure == 'aa':
            self.B = torch.nn.Parameter(torch.randn(self.n))
        elif network_structure == 'fc':
            self.B = torch.nn.Parameter(torch.randn(self.n, self.n))
        else:
            raise ValueError('Invalid network structure')

    def forward(self, afferent, prediction):
        B = (
            torch.sigmoid(self.B)
            if self.B.dim() == 2
            else create_agonist_antgonist_weight_matrix(torch.sigmoid(self.B))
        )
        I_p_B_inv = torch.linalg.inv(torch.eye(self.n) + B)
        activation = torch.matmul(afferent - prediction, I_p_B_inv)
        return torch.tanh(activation)

class IaReflex(nn.Module):
    def __init__(self, num_motor_units):
        super().__init__()
        self.n = num_motor_units
        self.K = torch.nn.Parameter(torch.randn(self.n))

    def forward(self, afferent, prediction, modulation):
        activation = (
            torch.sigmoid(self.K) * (afferent - prediction) - modulation
        )
        return torch.sigmoid(activation)

class SupraspinalDecoder(nn.Module):
    def __init__(self, num_encoding_features, num_motor_units):
        super().__init__()
        self.decoder = nn.Sequential(
                #nn.Linear(in_features,in_features),
                #nn.ReLU(),
                #nn.Linear(in_features,num_motor_units**2),
                #nn.ReLU(),
                #nn.Linear(num_motor_units**2,num_motor_units**2),
                #nn.ReLU(),
                #nn.Linear(num_motor_units**2,num_motor_units),
                #nn.ReLU(),
                #nn.Linear(num_motor_units,num_motor_units),
                #nn.ReLU(),
                nn.Linear(num_encoding_features,32),
                nn.Tanh(),
                nn.Linear(32,32),
                nn.Tanh(),
                nn.Linear(32,num_motor_units),
                nn.Tanh()
        )

    def forward(self, encoding_features):
        return self.decoder(encoding_features)





