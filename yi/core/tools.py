from .nodes import *
from ._tensor import *
import numpy as np
from .dtypes import float32


def tensor(data, dtype=float32, requires_grad=False, is_leaf=False):
    data = np.array(data, dtype)
    return Tensor(data, requires_grad=requires_grad, is_leaf=is_leaf)


def grad(y: Tensor, x: Tensor, dy=None, retain_graph=False):
    x.zero_grad()
    y.backward(dy, retain_graph=retain_graph)
    return x.grad


def backward(y: Tensor, dy=None, retain_graph=False):
    if y.grad_fn is None or not y.requires_grad:
        print(y.grad_fn, f"requires_grad={y.requires_grad}, 当前阶没有梯度或者梯度为0")
        return
    if dy is None:
        dy = np.full_like(y.data, 1., float32)
    if retain_graph:
        if not isinstance(dy, Tensor):
            dy = Tensor(dy, requires_grad=retain_graph)
        y.grad_fn.do_retain_backward(dy, retain_graph)
    else:
        y.grad_fn.do_backward(dy)


def clone(a: Tensor):
    new = object.__new__(Tensor)
    new.__dict__ = a.__dict__.copy()
    new.data = new.data.copy()
    return new


def add(a: Tensor, b: Tensor) -> Tensor: return Add(a, b)


def sub(a: Tensor, b: Tensor) -> Tensor: return Sub(a, b)


def mul(a: Tensor, b: Tensor) -> Tensor: return Mul(a, b)


def div(a: Tensor, b: Tensor) -> Tensor: return Div(a, b)


def pow(a: Tensor, b: Tensor) -> Tensor: return Pow(a, b)


def logn(a: Tensor, b: Tensor): return LogN(a, b)


def exp(a: Tensor) -> Tensor: return Exp.apply(a)


def log(a: Tensor) -> Tensor: return Log.apply(a)


def sin(a: Tensor) -> Tensor: return Sin.apply(a)


def cos(a: Tensor) -> Tensor: return Cos.apply(a)


def tan(a: Tensor) -> Tensor: return Tan.apply(a)


def asin(a: Tensor) -> Tensor: return ArcSin(a)


def acos(a: Tensor) -> Tensor: return ArcCos(a)


def atan(a: Tensor) -> Tensor: return ArcTan(a)


def transpose(a: Tensor, axes: Iterable[int]) -> Tensor: return Transpose(a, axes)


def abs(a: Tensor) -> Tensor: return Abs.apply(a)


def dot(a: Tensor, b: Tensor) -> Tensor: return Dot(a, b)


def sqrt(a: Tensor) -> Tensor: return Sqrt(a)


def reshape(a: Tensor, new_shape: Iterable[int]) -> Tensor: return Reshape(a, new_shape)


def cat(*tensors: Tensor, dim: int) -> Tensor: return Cat(*tensors, dim=dim)


concatenate = cat


def sum(a: Tensor, dim=None) -> Tensor: return Sum(a, dim)


def maximum(a: Tensor, b: Tensor) -> Tensor: return Maximum(a, b)


def minimum(a: Tensor, b: Tensor) -> Tensor: return Minimum(a, b)


def squeeze(a: Tensor, dim) -> Tensor: return Squeeze(a, dim)


def unsqueeze(a: Tensor, dim) -> Tensor: return UnSqueeze(a, dim)


def mean(a: Tensor, dim=None) -> Tensor: return Mean(a, dim)


def relu(a: Tensor) -> Tensor: return Relu.apply(a)


def flatten(a: Tensor, start_dim=0, end_dim=-1) -> Tensor:
    if end_dim < 0:
        end_dim = len(a.shape) + end_dim
    middle_shape = (-1,) if len(a.shape[start_dim: end_dim]) else ()
    new_shape = a.shape[:start_dim] + middle_shape + a.shape[end_dim + 1:]
    return reshape(a, new_shape)


def argmax(a: Tensor, dim=None) -> Tensor:
    return from_numpy(np.argmax(a.data, dim))


def argmin(a: Tensor, dim=None) -> Tensor:
    return from_numpy(np.argmin(a.data, dim))


def argsort(a: Tensor, dim=None) -> Tensor:
    return from_numpy(np.argsort(a.data, dim))


def argwhere(a: Tensor) -> Tensor:
    return from_numpy(np.argwhere(a.data))


def pad_constant(a: "Tensor", pad_width) -> Tensor:
    return PadConstant(a, pad_width)


def is_tensor(a: Any) -> bool:
    return isinstance(a, Tensor)


def dropout(a: Tensor, prob: float):
    return Dropout.apply(a, prob)
