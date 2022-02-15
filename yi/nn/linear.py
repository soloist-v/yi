from .base import Module
from ..core import Tensor, np, Add, dot
from .parameter import Parameter


class Identity(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input


class Linear(Module):
    def __init__(self, features_inp, features_out, bias=True):
        super().__init__()
        std = 1.0 / features_inp
        self.features_inp = features_inp
        self.features_out = features_out
        self.weights = Parameter(np.random.uniform(-std, std, (features_inp, features_out)))
        self.bias = Parameter(np.random.uniform(-std, std, features_out)) if bias else None

    def forward(self, x):
        if self.bias is None:
            return dot(x, self.weights)
        return Add(dot(x, self.weights), self.bias)

    def zero_grad(self):
        self.weights.zero_grad()
        if self.bias:
            self.bias.zero_grad()
