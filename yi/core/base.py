from typing import *
import numpy as np
from numpy import ndarray
from . import _tensor

if TYPE_CHECKING:
    from _tensor import Tensor


def filter_types(objs, types: Union[Type, Tuple[Type]]):
    return filter(lambda x: isinstance(x, types), objs)


class Context:
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def save_for_backward(self, *args):
        self["args"] = args

    @property
    def saved_tensors(self):
        return self['args']

    def __getitem__(self, item):
        return getattr(self, item)


class BaseBackwardFunction:

    def __init__(self, *inputs, **kwargs):
        """
        同时支持ndarray和Tensor运算
        :param inputs:
        :param kwargs:
        """
        self.kwargs = kwargs
        self.inputs: List["Tensor"] = []
        self.inputs_data: List[ndarray] = []
        for a in inputs:
            if not isinstance(a, (_tensor.Tensor, ndarray)):
                continue
            self.inputs.append(a)
            self.inputs_data.append(a.data)
        self.requires_grad = any(map(lambda _i: _i.requires_grad, self.inputs))

    def do_backward(self, dy):
        raise NotImplementedError("This method must be implement.")

    def do_retain_backward(self, dy, retain_grad):
        raise NotImplementedError("This node has no backward.")

    @classmethod
    def apply(cls, *args, **kwargs) -> "Tensor":
        cls: Type[Tensor]
        return cls(*args, **kwargs)


class ComputeNodeMeta(type):

    def __new__(mcs, what: str, bases: Tuple[Type], attr_dict: Dict[str, Any]):
        bases = tuple(t for t in bases if t not in (Backward,))
        cls = super().__new__(mcs, what, bases, attr_dict)
        backward_funcs = []
        retain_backward_funcs = []
        for name, v in cls.__dict__.items():
            if not callable(v):
                continue
            if name.startswith("backward"):
                backward_funcs.append(name)
            elif name.startswith("retain_backward"):
                retain_backward_funcs.append(name)
        backward_funcs.sort()
        retain_backward_funcs.sort()
        cls.backwards = backward_funcs
        cls.retain_backwards = retain_backward_funcs
        return cls

    def __call__(cls, *args, **kwargs) -> "Tensor":
        obj: ComputeNode = super().__call__(*args, **kwargs)
        res = obj.forward()
        return _tensor.Tensor(res, obj.requires_grad, obj if obj.requires_grad else None, is_leaf=False)


if not TYPE_CHECKING:
    class Backward:

        @overload
        def backward(self, dy: "ndarray") -> "ndarray":
            ...

        @overload
        def backward(self, dy: Union[int, float, "ndarray"]) -> "ndarray":
            ...

        def backward(self, dy=None) -> "ndarray":
            pass

        def retain_backward(self, dy: "Tensor") -> "Tensor":
            if not TYPE_CHECKING:
                raise NotImplementedError("This node has no backward.")
            return Tensor(0)
else:
    class Backward(_tensor.Tensor):

        @overload
        def backward(self, dy: "ndarray") -> "ndarray":
            ...

        @overload
        def backward(self, dy: Union[int, float, "ndarray"]) -> "ndarray":
            ...

        def backward(self, dy=None) -> "ndarray":
            pass

        def retain_backward(self, dy: "Tensor") -> "Tensor":
            if not TYPE_CHECKING:
                raise NotImplementedError("This node has no backward.")
            return Tensor(0)


# =====================================================================================================================
class ComputeNode(BaseBackwardFunction, Backward, metaclass=ComputeNodeMeta):

    def do_backward(self, dy):
        grads = self.backward(dy)
        grads = grads if isinstance(grads, (tuple, list)) else (grads,)  # GeneratorType
        for x, dx in zip(self.inputs, grads):
            assert x.shape == dx.shape, f"backward: {self}, x shape{x.shape}, dx shape: {dx.shape}"
            if x.requires_grad and x.grad_fn is None:
                x.grad += dx  # 对于叶子变量需要累加
            if x.grad_fn is not None:
                x.grad_fn.do_backward(dx)  # 中间梯度不保存，直接传递即可，报错会遇到非叶子节点梯度累加问题重复问题。

    def do_retain_backward(self, dy, retain_grad):
        grads = self.retain_backward(dy)
        grads = grads if isinstance(grads, (tuple, list)) else (grads,)  # GeneratorType
        for x, dx in zip(self.inputs, grads):
            assert x.shape == dx.shape, f"backward: {self}, x shape{x.shape}, dx shape: {dx.shape}"
            if x.requires_grad and x.grad_fn is None:
                x.grad += dx  # 对于叶子变量需要累加
                x.grad.requires_grad = retain_grad
            if x.grad_fn is not None:
                x.grad_fn.do_retain_backward(dx, retain_grad)  # 中间梯度不保存，直接传递即可，报错会遇到非叶子节点梯度累加问题重复问题。

    def forward(self) -> np.ndarray:
        raise NotImplementedError("This method must be implement.")

    def backward(self, dy: Union[int, "ndarray"] = None) -> Union[Tuple["ndarray", ...], "ndarray"]:
        """可以同时返回多个梯度或者一个都可以"""
        raise NotImplementedError("This method must be implement.")


# =====================================================================================================================
class ComputeNodeDetached(BaseBackwardFunction, Backward, metaclass=ComputeNodeMeta):
    """
    这个计算图是将输入tensor的梯度分离的
    反向传播计算的是输入的梯度，但是dy的shape是前向传播输出的shape，因为下一层就是这个shape，所以下一层的梯度也是这个shape
    注意这个类不能完全取代ComputeNode，因为这个类的backward只能返回一个梯度，不能同时返回两个梯度，梯度必须一一对应
    前向传播及时x在多个地方被使用，也不会改变x的值，因为每次x和其他tensor相乘都会生成新的tensor，并不会改变x的值，所以无需担心多个地方使用x
    时x中途被修改的问题，除非时写法上的问题。
    """
    backwards: List[str]
    retain_backwards: List[str]

    # 当前是计算节点，所以存储了所有输入变量的梯度计算函数
    # 前向传播及时x在多个地方被使用，也不会改变x的值，因为每次x和其他tensor相乘都会生成新的tensor，并不会改变x的值，所以无需担心多个地方使用x
    #     时x中途被修改的问题，除非时写法上的问题。
    def do_backward(self, dy):
        # 遍历当前计算节点的输入变量和变量对应的梯度计算函数，并计算其梯度
        for bfn_name, x in zip(self.backwards, self.inputs):
            if x.requires_grad:
                # 求梯度说明不是常量节点，常量不求梯度，但是当前节点有可能是叶子节点 没子节点没求给
                backward_fn = getattr(self, bfn_name)
                dx = backward_fn(dy)
                assert x.shape == dx.shape, f"backward: {self}<{backward_fn}>, x shape{x.shape}, dx shape: {dx.shape}"
                if x.grad_fn is None:
                    x.grad += dx  # 对于叶子变量需要累加
                else:
                    x.grad_fn.do_backward(dx)  # 中间梯度不保存，直接传递即可，报错会遇到非叶子节点梯度累加问题重复问题。

    def do_retain_backward(self, dy, retain_grad):
        # 遍历当前计算节点的输入变量和变量对应的梯度计算函数，并计算其梯度
        for bfn_name, x in zip(self.retain_backwards, self.inputs):
            if x.requires_grad:
                # 求梯度说明不是常量节点，常量不求梯度，但是当前节点有可能是叶子节点 没子节点没求给
                backward_fn = getattr(self, bfn_name)
                dx = backward_fn(dy)
                assert x.shape == dx.shape, f"backward: {self}<{backward_fn}>, x shape{x.shape}, dx shape: {dx.shape}"
                if x.grad_fn is None:
                    x.grad += dx  # 对于叶子变量需要累加
                    x.grad.requires_grad = retain_grad
                else:
                    x.grad_fn.do_retain_backward(dx, retain_grad)  # 中间梯度不保存，直接传递即可，报错会遇到非叶子节点梯度累加问题重复问题。

    def forward(self) -> "ndarray":
        if not TYPE_CHECKING:
            raise NotImplementedError("This method must be implement.")
        return np.ndarray()


# =====================================================================================================================
class FunctionMeta(type):
    def __init__(cls, name, classes, attr_kv):
        cls._backward_cls = type(name + "Backward", (BackwardFunction,), {"_backward_fn": cls})
        super().__init__(name, classes, attr_kv)

    def __call__(cls, *args, **kwargs):
        return getattr(cls, "apply")(*args, **kwargs)


class BackwardFunction(BaseBackwardFunction):
    _backward_fn: "Function"

    def __init__(self, ctx: Context, *inputs: "Tensor"):
        self.ctx = ctx
        super().__init__(*inputs)

    def do_backward(self, dy):
        grads = self._backward_fn.backward(self.ctx, dy)
        grads = grads if isinstance(grads, tuple) else (grads,)
        for x, dx in zip(self.inputs, grads):
            assert x.shape == dx.shape, f"backward: {self}, x shape{x.shape}, dx shape: {dx.shape}"
            if isinstance(dx, _tensor.Tensor):
                dx = dx.data
            # 如果grad是ndarray类型，grad.data是memoryview类型，
            # 而memoryview可以直接赋值给ndarray，速度和ndarray一致,类型也能正确匹配
            # 如果grad是Tensor，则grad.data恰好是ndarray，刚好直接赋值
            if x.requires_grad and x.grad_fn is None:
                x.grad += dx  # 对于叶子变量需要累加
            if x.grad_fn is not None:
                x.grad_fn.do_backward(dx)  # 中间梯度不保存，直接传递即可，报错会遇到非叶子节点梯度累加问题重复问题。

    def do_retain_backward(self, dy, retain_grad):
        grads = self._backward_fn.backward(self.ctx, dy)
        grads = grads if isinstance(grads, tuple) else (grads,)
        for x, dx in zip(self.inputs, grads):
            assert isinstance(dx, _tensor.Tensor), "backward function must be return a Tensor."
            assert x.shape == dx.shape, f"backward: {self}, x shape{x.shape}, dx shape: {dx.shape}"
            if x.requires_grad and x.grad_fn is None:
                x.grad += dx  # 对于叶子变量需要累加
                x.grad.requires_grad = retain_grad
            if x.grad_fn is not None:
                x.grad_fn.do_retain_backward(dx, retain_grad)  # 中间梯度不保存，直接传递即可，报错会遇到非叶子节点梯度累加问题重复问题。


class Function(metaclass=FunctionMeta):
    _backward_cls: Type[BackwardFunction]

    @classmethod
    def apply(cls, *inputs: "Tensor", **kwargs):
        ctx = Context()
        # res_data = cls.forward(ctx, *inputs, **kwargs)
        res_data = cls.forward(ctx, *(i.no_grad if isinstance(i, _tensor.Tensor) else i for i in inputs), **kwargs)
        requires_grad = any(map(lambda _i: _i.requires_grad, filter_types(inputs, _tensor.Tensor)))
        y = _tensor.Tensor(res_data, requires_grad, is_leaf=False)
        if requires_grad:
            y.grad_fn = cls._backward_cls(ctx, *inputs)
        return y

    @staticmethod
    def forward(ctx: Context, *args: "Tensor", **kwargs) -> np.ndarray:
        pass

    @staticmethod
    def backward(ctx: Context, dy: "Tensor") -> Union[Tuple["Tensor", ...], "Tensor"]:
        pass
