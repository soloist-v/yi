from ctypes import sizeof
from typing import *
from copy import copy
import numpy as np
from numpy import ndarray
from . import nodes
from . import tools
from . import dtypes

if TYPE_CHECKING:
    from .base import BaseBackwardFunction


def _tensor(data, dtype, requires_grad=False, is_leaf=False):
    data = np.array(data, dtype)
    return Tensor(data, requires_grad=requires_grad, is_leaf=is_leaf)


# 注意: 关于一个可能发生的bug是，如果一个变量多出被使用，但是它对应的data数组是一样的话，如果data被改变就会影响到其他地方


class Tensor:

    def __init__(self, data: Union[np.ndarray, int, float, List, Tuple, range, "Tensor"],
                 requires_grad=False, grad_fn: "BaseBackwardFunction" = None, is_leaf=True):
        if isinstance(data, Tensor):
            self.data: ndarray = data.data
        else:
            self.data: ndarray = data if isinstance(data, np.ndarray) else np.array(data, dtype=dtypes.float32)
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf and grad_fn is None
        self.grad = 0.
        self.grad_fn = grad_fn

    def requires_grad_(self, requires_grad):
        self.requires_grad = requires_grad
        return self

    def backward(self, dy=None, retain_graph=False):
        if self.grad_fn is None or not self.requires_grad:
            print(self.grad_fn, f"requires_grad={self.requires_grad}, 当前阶没有梯度或者梯度为0")
            return
        if dy is None:
            dy = np.full_like(self.data, 1., dtypes.float32)
        if retain_graph:
            if not isinstance(dy, Tensor):
                dy = Tensor(dy)
            dy.requires_grad = retain_graph
            self.grad_fn.do_retain_backward(dy, retain_graph)
        else:
            self.grad_fn.do_backward(dy)

    def zero_grad(self, set_to_none=False):
        if self.grad is None:
            return
        # self._grad.data[:] = 0.
        if set_to_none:
            self.grad = None
        else:
            self.grad = 0.

    def step(self, lr):
        if isinstance(self.grad, Tensor):
            self.data -= lr * self.grad.data
        else:
            self.data -= lr * self.grad

    # @property
    # def grad(self):
    #     return self._grad

    # @grad.setter
    # def grad(self, val):
    #     # assert val is None or isinstance(val, Tensor)
    #     self._grad = val

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def no_grad(self):
        if self.requires_grad:
            new = copy(self)
            new.requires_grad = False
            return new
        return self

    def size(self, dim=slice(None)):
        return self.data.shape[dim]

    def copy(self):
        new = object.__new__(Tensor)
        new.__dict__ = self.__dict__.copy()
        new.data = new.data.copy()
        new.is_leaf = False
        return new

    def copy_(self, data: Union['Tensor', ndarray]):
        if isinstance(data, Tensor):
            data = data.data
        self.data[:] = data

    def try_copy(self, data: Union['Tensor', ndarray]):
        if isinstance(data, Tensor):
            data = data.data
        t_data = data.reshape(-1, )
        this_data = self.data.reshape(-1, )
        len_min = min(len(this_data), len(t_data))
        this_data[: len_min] = t_data[: len_min]

    def clone(self):
        new = object.__new__(Tensor)
        new.__dict__ = self.__dict__.copy()
        new.data = new.data.copy()
        return new

    def detach(self):
        self.grad = None
        self.requires_grad = False
        return self

    def numpy(self):
        return self.data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data}, requires_grad={self.requires_grad}, " \
               f"shape={self.shape}, dtype={self.dtype}, grad_fn={self.grad_fn})"

    def __add__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return nodes.Add(self, b)

    def add_(self, b, alpha=None) -> "Tensor":
        if isinstance(b, Tensor):
            b = b.data
        if alpha:
            b = b * alpha
        self.data[:] += b
        return self

    def __radd__(self, b) -> "Tensor":
        return nodes.Add(_tensor(b, self.data.dtype), self)

    def __sub__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return nodes.Sub(self, b)

    def __rsub__(self, b) -> "Tensor":
        return nodes.Sub(_tensor(b, self.data.dtype), self)

    def __mul__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return nodes.Mul(self, b)

    def __rmul__(self, b) -> "Tensor":
        return nodes.Mul(_tensor(b, self.data.dtype), self)

    def __matmul__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return nodes.Dot(self, b)

    def __rmatmul__(self, b) -> "Tensor":
        return nodes.Dot(_tensor(b, self.data.dtype), self)

    def __truediv__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return nodes.Div(self, b)

    def __rtruediv__(self, b) -> "Tensor":
        return nodes.Div(_tensor(b, self.data.dtype), self)

    def __pow__(self, b, modulo=None) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return nodes.Pow(self, b)

    def __rpow__(self, b) -> "Tensor":
        return nodes.Pow(_tensor(b, self.data.dtype), self)

    def __neg__(self) -> "Tensor":
        return nodes.Mul(_tensor(-1, self.data.dtype), self)

    def __pos__(self) -> "Tensor":
        return nodes.Mul(_tensor(1, self.data.dtype), self)

    def __eq__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return Tensor(self.data == b.data)

    def __gt__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return Tensor(self.data > b.data)

    def __ge__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return Tensor(self.data >= b.data)

    def __lt__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return Tensor(self.data < b.data)

    def __le__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return Tensor(self.data <= b.data)

    def __ne__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return Tensor(self.data != b.data)

    def __and__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return Tensor(self.data & b.data)

    def __rand__(self, b) -> "Tensor":
        b = _tensor(b, self.data.dtype)
        return Tensor(self.data & b.data)

    def __or__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return Tensor(self.data | b.data)

    def __ror__(self, b) -> "Tensor":
        b = _tensor(b, self.data.dtype)
        return Tensor(self.data | b.data)

    def __xor__(self, b) -> "Tensor":
        if not isinstance(b, Tensor):
            b = _tensor(b, self.data.dtype)
        return Tensor(self.data ^ b.data)

    def __rxor__(self, b) -> "Tensor":
        b = _tensor(b, self.data.dtype)
        return Tensor(self.data ^ b.data)

    def __invert__(self) -> "Tensor":
        assert self.data.dtype == np.bool
        return Tensor(~self.data)

    def __getitem__(self, idx) -> "Tensor":
        if isinstance(idx, Tensor):
            idx = idx.data
        return nodes.Index(self, idx)

    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            value = value.data
        nodes.CopySlices(self, idx, value)

    def __delitem__(self, idx):
        del self.data[idx]
        if isinstance(self.grad, Tensor):
            del self.grad[idx]

    def __len__(self):
        return len(self.data)

    def __iter__(self) -> Iterator[ndarray]:
        """
        用于满足ndarray x tensor, arr 在左边时的计算返回一个numpy数组
        (此情况仅在Function的forward和backward中使用)
        如果想要返回tensor则必须满足tensor在左边.
        :return:
        """
        # print("__iter__")
        return iter(self.data)

    def iter(self) -> Iterator["Tensor"]:
        return iter([Tensor(d) for d in self.data])

    def __int__(self):
        return self.data.astype(int)

    def __float__(self):
        return self.data.astype(float)

    def __bool__(self):
        return self.data.astype(bool)

    def __sizeof__(self):
        return self.data.nbytes

    def __hash__(self):
        return id(self)

    @property
    def T(self) -> "Tensor":
        return nodes.T.apply(self)

    def mean(self, dim=None, keepdim=False) -> "Tensor":
        return nodes.Mean(self, dim, keepdim)

    def sum(self, dim=None, keepdim=False) -> "Tensor":
        return nodes.Sum(self, dim, keepdim)

    def reshape(self, *new_shape) -> "Tensor":
        if not isinstance(new_shape[0], int):
            new_shape = new_shape[0]
        return nodes.Reshape(self, new_shape)

    def view(self, *new_shape) -> "Tensor":
        if not isinstance(new_shape[0], int):
            new_shape = new_shape[0]
        return nodes.Reshape(self, new_shape)

    def dot(self, b) -> "Tensor":
        return nodes.Dot(self, b)

    def flatten(self, start_dim=0, end_dim=-1) -> "Tensor":
        # return graphs.Flatten(self, start_dim, end_dim)
        return tools.flatten(self, start_dim, end_dim)

    def transpose(self, *axes) -> 'Tensor':
        return nodes.Transpose(self, axes)

    def astype(self, dtype):
        return nodes.AsType(self, dtype)

    def permute(self, *axes) -> 'Tensor':
        return nodes.Transpose(self, axes)
