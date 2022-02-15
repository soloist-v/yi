"""
retain 用于实现高阶导数，分开实现是为了性能，毕竟大多数时候用不到高阶导数
需要注意的是 高阶导数并非所有的都实现了，有一些算子并未实现
"""
from typing import *
import numpy as np
from numpy import ndarray
from . import _tensor
from .base import ComputeNode, ComputeNodeDetached
from . import functional
from .dtypes import float32

if TYPE_CHECKING:
    from ._tensor import Tensor


def from_numpy(data: ndarray, requires_grad=False, is_leaf=False):
    return _tensor.Tensor(data, requires_grad=requires_grad, is_leaf=is_leaf)


def full_like(a: "Tensor", full_val: Union[int, float], dtype=None, requires_grad=False) -> "Tensor":
    data = np.full_like(a.data, full_val, dtype or a.dtype)
    return _tensor.Tensor(data, requires_grad=requires_grad)


def full(shape, fill_val, dtype=float32, requires_grad=False) -> "Tensor":
    data = np.full(shape, fill_val, dtype)
    return _tensor.Tensor(data, requires_grad=requires_grad)


def zeros(shape, dtype=float32, requires_grad=False) -> "Tensor":
    return _tensor.Tensor(np.zeros(shape, dtype), requires_grad=requires_grad)


def zeros_like(a: 'Tensor', dtype=None, requires_grad=False) -> "Tensor":
    return _tensor.Tensor(np.zeros_like(a.data, dtype or a.dtype), requires_grad=requires_grad)


def ones(shape, dtype=float32, requires_grad=False):
    return _tensor.Tensor(np.ones(shape, dtype), requires_grad=requires_grad)


def ones_like(a: "Tensor", dtype=None, requires_grad=False):
    return _tensor.Tensor(np.ones_like(a.data, dtype or a.dtype), requires_grad=requires_grad)


def apply_broadcast_a(sa, sb, grad_a):
    if sa == sb:
        return grad_a
    max_dim = max(len(sa), len(sb))
    new_sa = (1,) * (max_dim - len(sa)) + sa
    new_sb = (1,) * (max_dim - len(sb)) + sb
    for i, (da, db) in enumerate(zip(new_sa, new_sb)):
        if db != da == 1:
            grad_a = np.expand_dims(grad_a.sum(axis=i), i)
    return grad_a.reshape(sa)


def apply_broadcast_b(sa, sb, grad_b):
    if sa == sb:
        return grad_b
    max_dim = max(len(sa), len(sb))
    new_sa = (1,) * (max_dim - len(sa)) + sa
    new_sb = (1,) * (max_dim - len(sb)) + sb
    for i, (da, db) in enumerate(zip(new_sa, new_sb)):
        if da != db == 1:
            grad_b = np.expand_dims(grad_b.sum(axis=i), i)
    return grad_b.reshape(sb)


def apply_broadcast(sa, sb, grad_a, grad_b):
    if sa == sb:
        return grad_a, grad_b
    max_dim = max(len(sa), len(sb))
    new_sa = (1,) * (max_dim - len(sa)) + sa
    new_sb = (1,) * (max_dim - len(sb)) + sb
    for i, (da, db) in enumerate(zip(new_sa, new_sb)):
        if db != da == 1:
            # 这里求和的原因是广播产生了n条路径，根据乘法结合律，
            # 提取公共部分后就是当前节点的n条路径相加了
            # 例如 1 x 2 x 3 x c1 + 1 x 2 x 3 x c2 + 1 x 2 x 3 x c3 ==> 1 x 2 x 3 x (c1 + c2 + c3 + c4)
            grad_a = np.expand_dims(grad_a.sum(i), i)
        if da != db == 1:
            grad_b = np.expand_dims(grad_b.sum(i), i)
    return grad_a.reshape(sa), grad_b.reshape(sb)


def apply_broadcast_a_tensor(sa, sb, grad_a: "Tensor"):
    if sa == sb:
        return grad_a
    max_dim = max(len(sa), len(sb))
    new_sa = (1,) * (max_dim - len(sa)) + sa
    new_sb = (1,) * (max_dim - len(sb)) + sb
    for i, (da, db) in enumerate(zip(new_sa, new_sb)):
        if db != da == 1:
            # grad_a = np.expand_dims(grad_a.sum(axis=i), i)
            grad_a = UnSqueeze(grad_a.sum(dim=i), i)
    return grad_a.reshape(sa)


def apply_broadcast_b_tensor(sa, sb, grad_b: "Tensor"):
    if sa == sb:
        return grad_b
    max_dim = max(len(sa), len(sb))
    new_sa = (1,) * (max_dim - len(sa)) + sa
    new_sb = (1,) * (max_dim - len(sb)) + sb
    for i, (da, db) in enumerate(zip(new_sa, new_sb)):
        if da != db == 1:
            # grad_b = np.expand_dims(grad_b.sum(axis=i), i)
            grad_b = ExpandDims(grad_b.sum(dim=i), i)
    return grad_b.reshape(sb)


class Add(ComputeNodeDetached):
    def forward(self) -> ndarray:
        a, b = self.inputs_data
        return a + b

    def backward_a(self, dy):
        a, b = self.inputs_data
        res = apply_broadcast_a(a.shape, b.shape, dy)
        return res

    def backward_b(self, dy):
        a, b = self.inputs_data
        res = apply_broadcast_b(a.shape, b.shape, dy)
        return res

    def retain_backward_a(self, dy):
        a, b = self.inputs
        res = apply_broadcast_a_tensor(a.shape, b.shape, dy)
        return res

    def retain_backward_b(self, dy):
        a, b = self.inputs
        res = apply_broadcast_b_tensor(a.shape, b.shape, dy)
        return res

    @staticmethod
    def at(a: "Tensor", indices, b: "Tensor"):
        if not isinstance(b, (int, float)):
            b = (b,) * len(indices)
        if not isinstance(b, _tensor.Tensor):
            b = _tensor.Tensor(b)
        if isinstance(indices, list):
            for i, idx in enumerate(indices):
                a = a * 1
                a[idx] += b[i]
        elif isinstance(indices, tuple):
            a[indices] = b
        return a


class Sub(ComputeNodeDetached):
    def forward(self) -> ndarray:
        a, b = self.inputs_data
        return a - b

    def backward_a(self, dy):
        a, b = self.inputs_data
        return apply_broadcast_a(a.shape, b.shape, dy)

    def backward_b(self, dy):
        a, b = self.inputs_data
        dy = -dy
        return apply_broadcast_b(a.shape, b.shape, dy)

    def retain_backward_a(self, dy):
        a, b = self.inputs
        return apply_broadcast_a_tensor(a.shape, b.shape, dy)

    def retain_backward_b(self, dy):
        a, b = self.inputs
        dy = -dy
        return apply_broadcast_b_tensor(a.shape, b.shape, dy)


class Mul(ComputeNodeDetached):
    def forward(self) -> ndarray:
        a, b = self.inputs_data
        return a * b

    def backward_a(self, dy):
        a, b = self.inputs_data
        dy = b * dy
        return apply_broadcast_a(a.shape, b.shape, dy)

    def backward_b(self, dy):
        a, b = self.inputs_data
        dy = a * dy
        return apply_broadcast_b(a.shape, b.shape, dy)

    def retain_backward_a(self, dy):
        a, b = self.inputs
        dy = b * dy
        return apply_broadcast_a_tensor(a.shape, b.shape, dy)

    def retain_backward_b(self, dy):
        a, b = self.inputs
        dy = a * dy
        return apply_broadcast_b_tensor(a.shape, b.shape, dy)


class Div(ComputeNodeDetached):
    def forward(self) -> ndarray:
        a, b = self.inputs_data
        return a / b

    def backward_a(self, dy):
        a, b = self.inputs_data
        dy = (1 / b) * dy
        return apply_broadcast_a(a.shape, b.shape, dy)

    def backward_b(self, dy):
        a, b = self.inputs_data
        dy = -a * (b ** (-2)) * dy
        return apply_broadcast_b(a.shape, b.shape, dy)

    def retain_backward_a(self, dy):
        a, b = self.inputs
        dy = (1 / b) * dy
        return apply_broadcast_a_tensor(a.shape, b.shape, dy)

    def retain_backward_b(self, dy):
        a, b = self.inputs
        dy = -a * (b ** (-2)) * dy
        return apply_broadcast_b_tensor(a.shape, b.shape, dy)


class Pow(ComputeNodeDetached):
    def forward(self) -> ndarray:
        a, b = self.inputs_data
        return np.power(a, b)

    def backward_a(self, dy):
        a, b = self.inputs_data
        dy = b * (a ** (b - 1)) * dy
        return apply_broadcast_a(a.shape, b.shape, dy)

    def backward_b(self, dy):
        a, b = self.inputs_data
        dy = np.power(a, b) * np.log(a) * dy
        return apply_broadcast_b(a.shape, b.shape, dy)

    def retain_backward_a(self, dy):
        a, b = self.inputs
        dy = b * (a ** (b - 1)) * dy
        return apply_broadcast_a_tensor(a.shape, b.shape, dy)

    def retain_backward_b(self, dy):
        a, b = self.inputs
        dy = np.power(a, b) * np.log(a) * dy
        return apply_broadcast_b_tensor(a.shape, b.shape, dy)


class Exp(ComputeNode):

    def forward(self):
        a, = self.inputs_data
        res = np.exp(a)
        self.kwargs["res"] = res
        return res

    def backward(self, dy):
        res = self.kwargs['res']
        return res * dy

    def retain_backward(self, dy):
        res = _tensor.Tensor(self.kwargs['res'])
        return res * dy


class T(ComputeNode):
    def forward(self) -> ndarray:
        a, = self.inputs_data
        return a.T

    def backward(self, dy):
        return dy.T

    retain_backward = backward


class CopySlices(ComputeNode):
    def __init__(self, a: "Tensor", idx, val):
        super().__init__(a.copy())
        self.out = a
        self.idx = idx
        self.val = val

    def forward(self) -> ndarray:
        self.out.data[self.idx] = self.val
        self.out.grad_fn = self
        self.out.grad = None
        return self.out.data

    def backward(self, dy: Union[int, "ndarray"] = None) -> Union[Tuple["ndarray", ...], "ndarray"]:
        if self.out.is_leaf:
            raise RuntimeError(
                "变量已经移动到计算图内部了，移动到计算图内部的变量不能在之后进行修改，只能在移动之前修改.")
        dy[self.idx] = 0  # ComputeGraph是一次性计算完成的，没有分离，所以可以直接修改dy
        return dy

    retain_backward = backward


class AsType(ComputeNode):
    def __init__(self, a: "Tensor", dtype):
        super().__init__(a)
        self.new_dtype = dtype
        self.old_dtype = a.dtype

    def forward(self) -> np.ndarray:
        a, = self.inputs_data
        return a.astype(self.new_dtype)

    def backward(self, dy: Union[int, "ndarray"] = None) -> Union[Tuple["ndarray", ...], "ndarray"]:
        return dy.astype(self.old_dtype)

    def retain_backward(self, dy: "Tensor") -> "Tensor":
        return dy.astype(self.old_dtype)


class Swapaxes(ComputeNodeDetached):
    def __init__(self, a: "Tensor", axis1, axis2):
        super().__init__(a)
        self.axis1 = axis1
        self.axis2 = axis2

    def forward(self) -> ndarray:
        a, = self.inputs_data
        return np.swapaxes(a, self.axis1, self.axis2)

    def backward(self, dy: "ndarray") -> "ndarray":
        return np.swapaxes(dy, self.axis2, self.axis1)

    def retain_backward(self, dy: "Tensor") -> "Tensor":
        da = Swapaxes(dy, self.axis2, self.axis1)
        return da


class Dot(ComputeNodeDetached):
    """
    矩阵乘法不存在广播机制，因此很统一所以方便运算
    卷积可以转换为矩阵乘法im2col
    """

    def forward(self) -> ndarray:
        a, b = self.inputs_data
        res = np.dot(a, b)
        return res

    def backward_a(self, dy):
        _, b = self.inputs_data
        if b.ndim >= 2:
            b = np.swapaxes(b, -1, -2)
        res = np.dot(dy, b)
        for i in range(len(res.shape) - 2):
            res = res.sum(1)
        return res

    def backward_b(self, dy):
        a, _ = self.inputs_data
        if a.ndim >= 2:
            a = np.swapaxes(a, -1, -2)
        res = np.dot(a, dy)
        for i in range(len(res.shape) - 2):
            res = res.sum(1)
        return res

    def retain_backward_a(self, dy):
        _, b = self.inputs
        if b.ndim >= 2:
            b = Swapaxes(b, -1, -2)
        res = Dot(dy, b)
        for i in range(len(res.shape) - 2):
            res = res.sum(1)
        return res

    def retain_backward_b(self, dy):
        a, _ = self.inputs
        if a.ndim >= 2:
            a = Swapaxes(a, -1, -2)
        res = Dot(a, dy)
        for i in range(len(res.shape) - 2):
            res = res.sum(1)
        return res


class Abs(ComputeNode):

    def forward(self) -> ndarray:
        a, = self.inputs_data
        return np.abs(a)

    def backward(self, dy):
        a, = self.inputs_data
        mask = np.ones_like(dy)
        mask[a < 0] = -1
        return mask * dy

    def retain_backward(self, dy: "Tensor") -> "Tensor":
        a, = self.inputs
        mask = ones_like(dy)
        mask[a < 0] = -1
        return mask * dy


class Log(ComputeNode):

    def forward(self) -> ndarray:
        x, = self.inputs_data
        return np.log(x)

    def backward(self, dy: ndarray):
        x, = self.inputs_data
        grad = (1 / x)
        return grad * dy

    def retain_backward(self, dy: ndarray):
        x, = self.inputs
        grad = (1 / x)
        return grad * dy


class LogN(ComputeNodeDetached):
    def forward(self) -> ndarray:
        a, b = self.inputs_data
        return np.power(a, b)

    def backward_a(self, dy):
        a, b = self.inputs_data
        dy = (-1 / (((np.emath.logn(b, a)) ** 2) * a * np.log(b))) * dy
        return apply_broadcast_a(a.shape, b.shape, dy.astype(float32))

    def backward_b(self, dy):
        a, b = self.inputs_data
        dy = (1 / (b * np.log(a))) * dy
        return apply_broadcast_b(a.shape, b.shape, dy)

    def retain_backward_a(self, dy):
        a, b = self.inputs
        dy = (-1 / (((LogN(b, a)) ** 2) * a * Log(b))) * dy
        return apply_broadcast_a_tensor(a.shape, b.shape, dy)

    def retain_backward_b(self, dy):
        a, b = self.inputs
        dy = (1 / (b * Log(a))) * dy
        return apply_broadcast_b_tensor(a.shape, b.shape, dy)


class Sin(ComputeNode):

    def forward(self) -> ndarray:
        a, = self.inputs_data
        return np.sin(a)

    def backward(self, dy: ndarray):
        a, = self.inputs_data
        grad_a = np.cos(a)
        return grad_a * dy

    def retain_backward(self, dy: ndarray):
        a, = self.inputs
        grad_a = Cos(a)
        return grad_a * dy


class Cos(ComputeNode):
    def forward(self) -> ndarray:
        a, = self.inputs_data
        return np.cos(a)

    def backward(self, dy: ndarray):
        a, = self.inputs_data
        grad_a = -np.sin(a)
        return grad_a * dy

    def retain_backward(self, dy: ndarray):
        a, = self.inputs
        grad_a = -Sin(a)
        return grad_a * dy


class Tan(ComputeNode):
    def forward(self) -> ndarray:
        a, = self.inputs_data
        return np.tan(a)

    def backward(self, dy: ndarray):
        a, = self.inputs_data
        grad_a = 1 / (np.cos(a) ** 2)
        return grad_a * dy

    def retain_backward(self, dy: ndarray):
        a, = self.inputs
        grad_a = 1 / (Cos(a) ** 2)
        return grad_a * dy


class ArcSin(ComputeNodeDetached):
    def __init__(self, a):
        super().__init__(a)

    def forward(self) -> ndarray:
        a, = self.inputs_data
        return np.arcsin(a)

    def backward(self, dy: ndarray) -> ndarray:
        a = self.inputs_data[0]
        grad = 1 / np.sqrt(1 - a ** 2)
        return grad * dy

    def retain_backward(self, dy: ndarray) -> ndarray:
        a = self.inputs[0]
        grad = 1 / Sqrt(1 - a ** 2)
        return grad * dy


class ArcCos(ComputeNodeDetached):
    def __init__(self, a):
        super().__init__(a)

    def forward(self) -> ndarray:
        a, = self.inputs_data
        return np.arccos(a)

    def backward(self, dy: ndarray) -> ndarray:
        a, = self.inputs_data
        grad = -1 / np.sqrt(1 - a ** 2)
        return grad * dy

    def retain_backward(self, dy: ndarray) -> ndarray:
        a, = self.inputs
        grad = -1 / Sqrt(1 - a ** 2)
        return grad * dy


class ArcTan(ComputeNodeDetached):
    def __init__(self, a):
        super().__init__(a)

    def forward(self) -> ndarray:
        a, = self.inputs_data
        return np.arctan(a)

    def backward(self, dy: ndarray) -> ndarray:
        a, = self.inputs_data
        grad = 1 / (1 + a ** 2)
        return grad * dy

    def retain_backward(self, dy: ndarray) -> ndarray:
        a, = self.inputs
        grad = 1 / (1 + a ** 2)
        return grad * dy


class Transpose(ComputeNode):
    def __init__(self, a, axes):
        super().__init__(a)
        self.axes = axes

    def forward(self) -> ndarray:
        a = self.inputs_data[0]
        return np.transpose(a, self.axes)

    def backward(self, dy: Union[int, ndarray] = None) -> Union[Tuple[ndarray, ...], ndarray]:
        axes = list(range(len(self.axes)))
        for i, old_ax in enumerate(self.axes):
            axes[old_ax] = i  # 当前位置i对应的旧位置是old_i, 比如当前位置1，对应的旧位置3，所以应该把1放到3位置
        return np.transpose(dy, axes)

    def retain_backward(self, dy: Union[int, "Tensor"] = None) -> Union[Tuple["Tensor", ...], "Tensor"]:
        axes = list(range(len(self.axes)))
        for i, old_ax in enumerate(self.axes):
            axes[old_ax] = i  # 当前位置i对应的旧位置是old_i, 比如当前位置1，对应的旧位置3，所以应该把1放到3位置
        return Transpose(dy, axes)


class Index(ComputeNode):
    def __init__(self, a: "Tensor", idx: Union[ndarray, "Tensor"]):
        super().__init__(a)
        if isinstance(idx, _tensor.Tensor):
            idx = idx.data
        self.idx = idx

    def forward(self) -> ndarray:
        return self.inputs_data[0][self.idx]

    def backward(self, dy: Union[int, ndarray] = None) -> Union[Tuple[ndarray, ...], ndarray]:
        a, = self.inputs_data
        grad = np.zeros_like(a)
        grad[self.idx] = dy
        return grad

    def retain_backward(self, dy: Union[int, "Tensor"] = None) -> Union[Tuple["Tensor", ...], "Tensor"]:
        a, = self.inputs
        grad = zeros_like(a)
        grad[self.idx] = dy
        return grad


class Mean(ComputeNode):
    def __init__(self, a, dim=None, keepdim=False):
        super().__init__(a)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self) -> ndarray:
        a, = self.inputs_data
        return np.mean(a, axis=self.dim, keepdims=self.keepdim)

    def backward(self, dy: Union[int, ndarray] = None) -> Union[Tuple[ndarray, ...], ndarray]:
        a, = self.inputs_data
        if self.dim is None:
            val = 1 / int(np.prod(a.shape))
        else:
            if isinstance(self.dim, int):
                val = 1 / a.shape[self.dim]
                dims = (self.dim,)
            else:
                dims = self.dim
                val = 1
                for dim in dims:
                    val *= 1 / a.shape[dim]
            if not self.keepdim:
                for dim in sorted(dims):
                    dy = np.expand_dims(dy, dim)
        return np.full(a.shape, val, float32) * dy

    def retain_backward(self, dy: Union[int, "Tensor"] = None) -> Union[Tuple["Tensor", ...], "Tensor"]:
        a, = self.inputs
        if self.dim is None:
            val = 1 / int(np.prod(a.shape))
        else:
            val = 1 / a.shape[self.dim]
            if not self.keepdim:
                dy = ExpandDims(dy, self.dim)
        return full_like(a, val) * dy


class Sqrt(ComputeNode):

    def forward(self) -> ndarray:
        a, = self.inputs_data
        return np.sqrt(a)

    def backward(self, dy: Union[int, ndarray] = None) -> Union[Tuple[ndarray, ...], ndarray]:
        a, = self.inputs_data
        # x^0.5 ==> 0.5 * x^(0.5 - 1) ==> 0.5 * (1 / x^0.5)
        grad = 1 / (2 * np.sqrt(a))
        return grad * dy

    def retain_backward(self, dy: Union[int, ndarray] = None) -> Union[Tuple[ndarray, ...], ndarray]:
        a, = self.inputs
        # x^0.5 ==> 0.5 * x^(0.5 - 1) ==> 0.5 * (1 / x^0.5)
        grad = 1 / (2 * Sqrt(a))
        return grad * dy


class Sum(ComputeNode):

    def __init__(self, a: "Tensor", dim=None, keepdim=False):
        """
        if dim is None, a will be flatten first.
        :param a:
        :param dim:
        """
        super().__init__(a)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self) -> ndarray:
        a, = self.inputs_data
        return np.sum(a, axis=self.dim, keepdims=self.keepdim)

    def backward(self, dy: Union[ndarray, int, float] = None):
        a, = self.inputs_data
        if self.dim is None:
            return np.ones_like(a) * dy
        if not self.keepdim:
            dy = np.expand_dims(dy, self.dim)
        return np.ones_like(a) * dy

    def retain_backward(self, dy: Union["Tensor", int, float] = None):
        a, = self.inputs
        if self.dim is None:
            return ones_like(a) * dy
        if not self.keepdim:
            dy = ExpandDims(dy, self.dim)
        return ones_like(a) * dy


class Cat(ComputeNode):

    def __init__(self, *tensors, dim):
        super().__init__(*tensors)
        assert len(tensors), "At least one tensor"
        self.dim = dim if dim is not None else len(tensors[0].shape) - 1

    def forward(self) -> ndarray:
        return np.concatenate(self.inputs_data, axis=self.dim)

    def backward(self, dy):
        start = 0
        grads = []
        for v in self.inputs:
            shape = v.shape
            s = tuple(slice(start, start + shape[self.dim]) if i == self.dim else ... for i in range(self.dim + 1))
            grads.append(dy[s])
            start += shape[self.dim]
        return grads

    retain_backward = backward


class Reshape(ComputeNodeDetached):
    """
    像这种什么都没干的操作在计算图中相当于 y=x, dy/dx = 1, backward => dy * 1 = dy
    """

    def __init__(self, a: "Tensor", new_shape):
        super().__init__(a)
        self.new_shape = new_shape
        self.src_shape = a.shape

    def forward(self) -> ndarray:
        a, = self.inputs_data
        res = np.reshape(a, self.new_shape)
        return res

    def backward(self, dy: ndarray = None) -> ndarray:
        dy = np.reshape(dy, self.src_shape)
        return dy

    def retain_backward(self, dy: "Tensor" = None) -> "Tensor":
        return Reshape(dy, self.src_shape)


class Maximum(ComputeNodeDetached):
    def __init__(self, a, b):
        super().__init__(a, b)

    def forward(self) -> ndarray:
        a, b = self.inputs_data
        return np.maximum(a, b)

    def backward_a(self, dy: Union[int, ndarray] = None) -> ndarray:
        a, b = self.inputs_data
        mask_a = a > b
        grad_a = np.zeros_like(a)
        grad_a[mask_a] = dy[mask_a]
        return apply_broadcast_a(a.shape, b.shape, grad_a)

    def backward_b(self, dy: Union[int, ndarray] = None) -> ndarray:
        a, b = self.inputs_data
        mask_b = b > a
        grad_b = np.zeros_like(b)
        grad_b[mask_b] = dy[mask_b]
        return apply_broadcast_b(a.shape, b.shape, grad_b)

    def retain_backward_a(self, dy: Union[int, "Tensor"] = None) -> "Tensor":
        a, b = self.inputs
        mask_a = a > b
        grad_a = zeros_like(a)
        grad_a[mask_a] = dy[mask_a]
        return apply_broadcast_a_tensor(a.shape, b.shape, grad_a)

    def retain_backward_b(self, dy: Union[int, "Tensor"] = None) -> "Tensor":
        a, b = self.inputs
        mask_b = b > a
        grad_b = zeros_like(b)
        grad_b[mask_b] = dy[mask_b]
        return apply_broadcast_b_tensor(a.shape, b.shape, grad_b)


class Minimum(ComputeNodeDetached):
    def __init__(self, a: "Tensor", b: "Tensor"):
        super().__init__(a, b)

    def forward(self) -> ndarray:
        a, b = self.inputs_data
        return np.minimum(a, b)

    def backward_a(self, dy: Union[int, ndarray] = None) -> ndarray:
        a, b = self.inputs_data
        mask_a = a < b
        grad_a = np.zeros_like(a)
        grad_a[mask_a] = dy[mask_a]
        return apply_broadcast_a(a.shape, b.shape, grad_a)

    def backward_b(self, dy: Union[int, ndarray] = None) -> ndarray:
        a, b = self.inputs_data
        mask_b = b < a
        grad_b = np.zeros_like(b)
        grad_b[mask_b] = dy[mask_b]
        return apply_broadcast_b(a.shape, b.shape, grad_b)

    def retain_backward_a(self, dy: Union[int, "Tensor"] = None) -> "Tensor":
        a, b = self.inputs
        mask_a = a < b
        grad_a = zeros_like(a)
        grad_a[mask_a] = dy[mask_a]
        return apply_broadcast_a_tensor(a.shape, b.shape, grad_a)

    def retain_backward_b(self, dy: Union[int, "Tensor"] = None) -> "Tensor":
        a, b = self.inputs
        mask_b = b < a
        grad_b = zeros_like(b)
        grad_b[mask_b] = dy[mask_b]
        return apply_broadcast_b_tensor(a.shape, b.shape, grad_b)


class UnSqueeze(ComputeNode):
    def __init__(self, a: "Tensor", dim):
        super().__init__(a)
        self.dim = dim

    def forward(self) -> ndarray:
        a = self.inputs_data[0]
        return np.expand_dims(a, self.dim)

    def backward(self, dy: Union[int, ndarray] = None) -> Union[Tuple[ndarray, ...], ndarray]:
        return np.squeeze(dy, self.dim)

    def retain_backward(self, dy: Union[int, "Tensor"] = None) -> Union[Tuple["Tensor", ...], "Tensor"]:
        return Squeeze(dy, self.dim)


ExpandDims = UnSqueeze


def expand_dims(a: "Tensor", dim):
    return UnSqueeze(a, dim)


class Squeeze(ComputeNode):
    def __init__(self, a: "Tensor", dim):
        super().__init__(a)
        self.dim = dim

    def forward(self) -> ndarray:
        a = self.inputs_data[0]
        return np.squeeze(a, self.dim)

    def backward(self, dy: Union[int, ndarray] = None) -> Union[Tuple[ndarray, ...], ndarray]:
        return np.expand_dims(dy, self.dim)

    def retain_backward(self, dy: Union[int, "Tensor"] = None) -> Union[Tuple["Tensor", ...], "Tensor"]:
        return UnSqueeze(dy, self.dim)


class Relu(ComputeNode):
    def forward(self):
        a, = self.inputs_data
        res = np.maximum(a, 0)
        return res

    def backward(self, dy):
        a, = self.inputs_data
        grad_a = np.zeros_like(a)
        mask = a > 0
        grad_a[mask] = dy[mask]
        return grad_a

    def retain_backward(self, dy):
        a, = self.inputs
        grad_a = zeros_like(a)
        mask = a.data > 0  # mask是ndarray就可以
        grad_a[mask] = dy[mask]
        return grad_a


class Flatten(ComputeNode):
    def __init__(self, a: "Tensor", start_dim=0, end_dim=-1):
        super().__init__(a)
        self.start_dim = start_dim
        if end_dim < 0:
            end_dim = len(a.shape) + end_dim
        self.end_dim = end_dim
        self.middle_shape = (-1,) if len(a.shape[start_dim: end_dim]) else ()

    def forward(self) -> ndarray:
        a = self.inputs_data[0]
        new_shape = a.shape[:self.start_dim] + self.middle_shape + a.shape[self.end_dim + 1:]
        return np.reshape(a, new_shape)

    def backward(self, dy: Union[int, ndarray] = None) -> Union[Tuple[ndarray, ...], ndarray]:
        a, = self.inputs_data
        return dy.reshape(a.shape)

    def retain_backward(self, dy: Union[int, "Tensor"] = None) -> Union[Tuple["Tensor", ...], "Tensor"]:
        a, = self.inputs
        return dy.reshape(a.shape)


class _Split(ComputeNodeDetached):

    def __init__(self, a: "Tensor", part: ndarray, idx):
        super().__init__(a)
        self.part = part
        self.idx = idx

    def forward(self) -> ndarray:
        return self.part

    def backward(self, dy):
        a, = self.inputs_data
        grad = np.zeros_like(a)  # 其他位置的数并未参与运算，所以对于当前这批数据来说梯度始终为0
        grad[self.idx] = dy
        return grad

    def retain_backward(self, dy):
        a, = self.inputs
        grad = zeros_like(a)  # 其他位置的数并未参与运算，所以对于当前这批数据来说梯度始终为0
        grad[self.idx] = dy
        return grad


def split(a: "Tensor", indices_or_sections, dim=0) -> List["Tensor"]:
    parts = np.split(a.data, indices_or_sections, dim)
    cur_i = 0
    res = []
    for part in parts:
        dim_len = part.shape[dim]
        start_i = cur_i
        end_i = cur_i + dim_len
        idx = [slice(None) for i in range(len(a.shape))]
        idx[dim] = slice(start_i, end_i)
        idx = tuple(idx)
        res.append(_Split(a, part, idx))
        cur_i = end_i
    return res


def dsplit(a, indices_or_sections):
    return split(a, indices_or_sections, 2)


def hsplit(a, indices_or_sections):
    return split(a, indices_or_sections, 1)


def vsplit(a, indices_or_sections):
    return split(a, indices_or_sections, 0)


class PadConstant(ComputeNode):
    def __init__(self, a: "Tensor", pad_width, constant_value=0):
        super().__init__(a)
        self.pad_width = pad_width
        self.mode = "constant"
        self.constant_values = constant_value
        self.idx = None
        assert len(pad_width) == len(a.shape)

    def forward(self) -> ndarray:
        a, = self.inputs_data
        res = np.pad(a, self.pad_width, self.mode, constant_values=self.constant_values)
        return res

    def backward(self, dy: Union[int, ndarray] = None) -> Union[Tuple[ndarray, ...], ndarray]:
        # (0, 0), (0, 0), (pad, pad), (pad, pad)
        idx = tuple(slice(pad_start, -pad_end) for pad_start, pad_end in self.pad_width)
        return dy[idx]

    def retain_backward(self, dy: Union[int, "Tensor"] = None) -> Union[Tuple["Tensor", ...], "Tensor"]:
        # (0, 0), (0, 0), (pad, pad), (pad, pad)
        idx = tuple(slice(pad_start, -pad_end) for pad_start, pad_end in self.pad_width)
        return dy[idx]


class Dropout(ComputeNode):

    def __init__(self, a: "Tensor", prob=0.5):
        super().__init__(a)
        self.prob = prob
        self.mask = None

    def forward(self):
        x, = self.inputs_data
        self.mask = np.random.binomial(1, self.prob, size=x.shape).astype("float32") / self.prob
        out = x * self.mask
        return out.reshape(x.shape)

    def backward(self, dy):
        return dy * self.mask

    def retain_backward(self, dy: "Tensor") -> "Tensor":
        return dy * _tensor.Tensor(self.mask)


class Im2Col(ComputeNode):
    def __init__(self, a, field_height=3, field_width=3, stride=1, padding=1):
        super().__init__(a)
        self.field_height = field_height
        self.field_width = field_width
        self.padding = padding
        self.stride = stride

    def forward(self) -> ndarray:
        a, = self.inputs_data
        cols = functional.im2col_indices(a, self.field_height, self.field_width, self.padding, self.stride)
        return cols

    def backward(self, dy: Union[int, ndarray] = None) -> Union[Tuple[ndarray, ...], ndarray]:
        x, = self.inputs_data
        dx = functional.col2im_indices(dy, x.shape, self.field_height, self.field_width, self.padding, self.stride)
        return dx


class Max(ComputeNode):
    def __init__(self, a: "Tensor", dim=None):
        super().__init__(a)
        assert dim is None or isinstance(dim, int), f"the dim must be a int."
        self.dim = dim
        self.keep_dim = False

    def forward(self) -> np.ndarray:
        a, = self.inputs_data
        return np.max(a, axis=self.dim)

    def backward(self, dy: Union[int, "ndarray"] = None) -> Union[Tuple["ndarray", ...], "ndarray"]:
        """当前实现不完善"""
        a, = self.inputs_data
        da = np.zeros_like(a)
        indices = np.argmax(a, axis=self.dim)
        if self.dim is None:
            da.reshape(-1, )[indices] = dy
        elif self.keep_dim:
            da[indices] = dy[self.dim]
        else:
            da[indices] = dy
        return da


class Min(ComputeNode):
    def __init__(self, a: "Tensor", dim=None):
        super().__init__(a)
        self.dim = dim
        self.keep_dim = False

    def forward(self) -> np.ndarray:
        a, = self.inputs_data
        return np.min(a, axis=self.dim)

    def backward(self, dy: Union[int, "ndarray"] = None) -> Union[Tuple["ndarray", ...], "ndarray"]:
        """当前实现不完善"""
        a, = self.inputs_data
        da = np.zeros_like(a)
        indices = np.argmin(a, axis=self.dim)
        if self.dim is None:
            da.reshape(-1, )[indices] = dy
        elif self.keep_dim:
            da[indices] = dy[self.dim]
        else:
            da[indices] = dy
        return da
