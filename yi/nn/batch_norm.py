from .base import Module
from ..core import Tensor, sqrt, np, float32, tensor
from .parameter import Parameter


def batch_norm(x: Tensor, gamma, beta, moving_mean, moving_var, eps, momentum, train=True):
    if not train:
        x_hat = (x - moving_mean) / sqrt(moving_var + eps)
    else:
        assert len(x.shape) in (2, 4)
        if x.ndim == 2:
            mean = x.mean(dim=0)
            var = ((x - mean) ** 2).mean(dim=0)
        else:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        x_hat = (x - mean) / sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    y = gamma * x_hat + beta
    return y, moving_mean, moving_var


class BatchNorm(Module):
    def __init__(self, num_features, num_dims, eps=1e-5, momentum=0.9):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = Parameter(np.ones(shape, float32))
        self.beta = Parameter(np.zeros(shape, float32))
        self.moving_mean = tensor(np.zeros(shape, float32))
        self.moving_var = tensor(np.ones(shape, float32))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

    def forward(self, x: Tensor):
        y, self.moving_mean, self.moving_var = batch_norm(
            x, self.gamma, self.beta, self.moving_mean, self.moving_var, self.eps, self.momentum, self.training)
        return y


class BatchNorm1d(BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__(num_features, 2, eps, momentum)


class BatchNorm2d(BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__(num_features, 4, eps, momentum)
