from .core import *
from .nn import Parameter
from . import nn


def relu_(x: Tensor) -> Tensor:
    return Maximum(x, Tensor(np.zeros(x.shape, np.float)))


def relu(a: Tensor) -> Tensor: return Relu.apply(a)


def sigmoid(x):
    """
    sigmoid = 1 / (1 + exp(-x))
    :param x:
    :return:
    """
    return 1 / (1 + Exp.apply(-x))


def tanh(x: Tensor):
    E = np.e
    return (E ** x - E ** (-x)) / (E ** x + E ** (-x))


def softplus(x: Tensor) -> Tensor:
    return Log.apply(1 + Exp.apply(x))


def mish(x: Tensor):
    return x * tanh(softplus(x))


def silu(x: Tensor):
    """
    效果挺好的，跟mish差不多，但是速度更快
    :param x:
    :return:
    """
    return x * sigmoid(x)


def hardtanh(x: Tensor, min_val=-1., max_val=1.) -> Tensor:
    return Maximum(Minimum(x, Tensor(max_val)), Tensor(min_val))


def hardswish(x: Tensor) -> Tensor:
    return x * hardtanh(x + 3, 0., 6.) / 6.


def leakyrelu(x: Tensor, negative_slope: float = 1e-2) -> Tensor:
    f1 = 0.5 * (1 + negative_slope)
    f2 = 0.5 * (1 - negative_slope)
    return f1 * x + f2 * abs(x)


class MemoryEfficientSwish(nn.Module):
    class F(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):
        return self.F.apply(x)


class MemoryEfficientMish(nn.Module):
    class F(Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * (tanh(softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = sigmoid(x)
            fx = tanh(softplus(x))
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


class AconC(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()
        self.p1 = Parameter(np.random.randn(1, in_channels))
        self.p2 = Parameter(np.random.randn(1, in_channels))
        self.beta = Parameter(np.ones((1, in_channels), float))

    def forward(self, x):
        return (self.p1 * x - self.p2 * x) * sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x


class PReLU(nn.Module):
    def __init__(self, channels=1, a=0.25):
        super().__init__()
        if channels == 1:
            self.a = Parameter(a)
        else:
            self.a = Parameter([a] * channels)

    def forward(self, x: Tensor) -> Tensor:
        return Maximum(x, x * self.a)


if __name__ == '__main__':
    x = tensor([1.], requires_grad=True)
    y = MemoryEfficientSwish()(x)
    print(y)
    y.backward()
    print(x.grad)
    exit()
    x = tensor(np.array([-2, 0, 2]), dtype=float32, requires_grad=True)
    # res = mish(x)
    res = relu(x)
    res.backward()
    print(res)
    print(x.grad)
