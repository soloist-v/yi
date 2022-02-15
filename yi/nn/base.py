from collections import OrderedDict
import numpy as np
from typing import Union, Dict, TYPE_CHECKING
from ..core import Tensor
from warnings import warn

if TYPE_CHECKING:
    from .parameter import Parameter
"""
tip: 构建计算图只需要关注Tensor"操作"或"变换"即可，因为只有Tensor的变换操作才会产生计算图;
     是否需要求导的条件是: 有没有数参与了当前或者之后的运算, 如果有那就需要对参与的数进行求导;
pytorch 不允许对需要求解梯度的变量使用in-place 操作，也就是setitem操作，这些操作没有产生计算图.
"""


def getattr_str(obj, attr_str: str):
    attrs = attr_str.split('.')
    last_obj = obj
    for attr in attrs:
        last_obj = getattr(last_obj, attr)
    return last_obj


def setattr_str(obj, attr_str: str, value):
    attrs = attr_str.split(".")
    last_obj = obj
    for attr in attrs[:-1]:
        last_obj = getattr(last_obj, attr)
    setattr(last_obj, attrs[-1], value)


class Module:
    __counter__ = 0

    def __init__(self):
        self._parameters: Dict[str, Union[Tensor, "Parameter"]] = OrderedDict()
        self._modules: Dict[str, Module] = OrderedDict()
        self.training = True
        self._id = self.__counter__
        Module.__counter__ += 1

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def __hash__(self):
        return id(self)

    def __setattr__(self, key, value):
        if isinstance(value, Tensor):
            if value.requires_grad:
                self._parameters[key] = value
        elif isinstance(value, Module):
            self._modules[key] = value
        super().__setattr__(key, value)

    def __delattr__(self, item):
        if self._modules.pop(item, None) is None:
            self._parameters.pop(item, None)
        super().__delattr__(item)

    def forward(self, *args, **kwargs) -> Tensor:
        pass

    def walk_parameters(self, prefix=""):
        for key, param in self._parameters.items():
            yield f'{prefix}{key}', param
        for key, m in self._modules.items():
            yield from m.walk_parameters(f"{prefix}{key}.")

    def walk_modules(self, prefix=""):
        for key, m in self._modules.items():
            yield f'{prefix}{key}', m
            yield from m.walk_modules(f"{prefix}{key}.")

    named_parameters = walk_parameters
    named_modules = walk_modules

    def parameters(self):
        return [param for _, param in self.walk_parameters()]

    def modules(self):
        for name, m in self.walk_modules():
            yield m

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        for param in self._parameters.values():
            param.requires_grad = self.training

    def eval(self):
        self.train(False)

    def random_parameters(self):
        for param in self._parameters.values():
            if isinstance(param, Tensor):
                param.data[:] = np.random.rand(*param.shape)
            elif isinstance(param, Module):
                param.random_parameters()

    def state_dict(self):
        params = OrderedDict()
        for key, param in self.walk_parameters():
            params[key] = param
        return params

    def load_state_dict(self, state_dict: Dict['str', Union[Tensor]],
                        strict: bool = True):
        param: Tensor
        for attr_str, value in state_dict.items():
            try:
                param = getattr_str(self, attr_str)
                param.try_copy(value)
            except KeyError as e:
                if strict:
                    raise e
                warn(f"{e}, {attr_str} load failed.", Warning)

    def step(self, lr):
        for param in self._parameters.values():
            param.step(lr)
        for m in self._modules.values():
            m.step(lr)

    def zero_grad(self):
        for param in self._parameters.values():
            param.zero_grad()
        for m in self._modules.values():
            m.zero_grad()

    def add_module(self, name, m: "Module"):
        setattr(self, name, m)

    def register_parameter(self, name, param):
        setattr(self, name, param)

    def requires_grad_(self, val):
        for param in self._parameters.values():
            param.requires_grad_(val)
        for m in self._modules.values():
            m.requires_grad_(val)
        return self
