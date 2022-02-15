from .base import Module
from ..core import Tensor, exp, relu, Max, sum


class Softmax(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        e_x = exp(x - Max(x))
        return e_x / (sum(e_x, dim=1)[:, None])


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.one = Tensor(1, False)

    def forward(self, x: Tensor):
        return 1 / (1 + exp(-x))


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return relu(x)
