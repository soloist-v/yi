import warnings
from collections import OrderedDict
from typing import Optional, Iterable, Iterator, TypeVar, Dict, Union, Callable, ItemsView, Mapping, overload, Any
from .base import Module
from ..core import Tensor
from .parameter import Parameter

T = TypeVar('T')


class FunctionWrapper(Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def __repr__(self):
        return f"{self.func}"

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class Sequential(Module):

    def __init__(self, *modules: Union[Module, Callable]):
        super().__init__()
        for i, m in enumerate(modules):
            if not isinstance(m, Module):
                if callable(m):
                    warnings.warn(f"{m} is not a Module subclass", UserWarning)
                else:
                    raise TypeError(f"{m} not in (Module, Callable)")
                m = FunctionWrapper(m)
            self.add_module(str(i), m)

    def forward(self, x: Tensor) -> Tensor:
        for m in self._modules.values():
            x = m(x)
        return x


class ModuleList(Module):

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def __getitem__(self, idx: int) -> Module:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[str(idx)]

    def __setitem__(self, idx: int, module: Module) -> None:
        return setattr(self, str(idx), module)

    def __delitem__(self, idx: Union[int, slice]) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, str(idx))

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __iadd__(self, modules: Iterable[Module]):
        return self.extend(modules)

    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self: T, index: int, module: Module) -> None:
        r"""Insert a given module before a given index in the list.

        Arguments:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self: T, module: Module) -> T:
        r"""Appends a given module to the end of the list.

        Arguments:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> T:
        r"""Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self


class ModuleDict(Module):

    def __init__(self, modules: Optional[Dict[str, Module]] = None) -> None:
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self) -> None:
        """Remove all items from the ModuleDict.
        """
        self._modules.clear()

    def pop(self, key: str) -> Module:
        v = self[key]
        del self[key]
        return v

    def keys(self) -> Iterable[str]:
        return self._modules.keys()

    def items(self) -> ItemsView[str, Module]:
        return self._modules.items()

    def values(self) -> Iterable[Module]:
        r"""Return an iterable of the ModuleDict values.
        """
        return self._modules.values()

    def update(self, modules: Mapping[str, Module]) -> None:
        if isinstance(modules, (OrderedDict, ModuleDict, Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            for key, module in modules:
                self[key] = module


class ParameterList(Module):
    def __init__(self, parameters: Optional[Iterable['Parameter']] = None) -> None:
        super(ParameterList, self).__init__()
        if parameters is not None:
            self.extend(parameters)

    @overload
    def __getitem__(self, idx: int) -> 'Parameter':
        ...

    @overload
    def __getitem__(self: T, idx: slice) -> T:
        ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._parameters.values())[idx])
        else:
            return self._parameters[str(idx)]

    def __setitem__(self, idx: int, param: 'Parameter') -> None:
        return self.register_parameter(str(idx), param)

    def __setattr__(self, key: Any, value: Any) -> None:
        if not isinstance(value, Parameter):
            warnings.warn("Setting attributes on ParameterList is not supported.")
        super(ParameterList, self).__setattr__(key, value)

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator['Parameter']:
        return iter(self._parameters.values())

    def __iadd__(self: T, parameters: Iterable['Parameter']) -> T:
        return self.extend(parameters)

    def __dir__(self):
        keys = super(ParameterList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self: T, parameter: 'Parameter') -> T:
        self.register_parameter(str(len(self)), parameter)
        return self

    def extend(self: T, parameters: Iterable['Parameter']) -> T:
        if not isinstance(parameters, Iterable):
            raise TypeError("ParameterList.extend should be called with an "
                            "iterable, but got " + type(parameters).__name__)
        offset = len(self)
        for i, param in enumerate(parameters):
            self.register_parameter(str(offset + i), param)
        return self


class ParameterDict(Module):

    def __init__(self, parameters: Optional[Mapping[str, 'Parameter']] = None) -> None:
        super(ParameterDict, self).__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key: str) -> 'Parameter':
        return self._parameters[key]

    def __setitem__(self, key: str, parameter: 'Parameter') -> None:
        self.register_parameter(key, parameter)

    def __delitem__(self, key: str) -> None:
        del self._parameters[key]

    def __setattr__(self, key: Any, value: Any) -> None:
        if not isinstance(value, Parameter):
            warnings.warn("Setting attributes on ParameterDict is not supported.")
        super(ParameterDict, self).__setattr__(key, value)

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._parameters

    def clear(self) -> None:
        """Remove all items from the ParameterDict.
        """
        self._parameters.clear()

    def pop(self, key: str) -> 'Parameter':
        r"""Remove key from the ParameterDict and return its parameter.

        Arguments:
            key (string): key to pop from the ParameterDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self) -> Iterable[str]:
        r"""Return an iterable of the ParameterDict keys.
        """
        return self._parameters.keys()

    def items(self) -> ItemsView[str, Union[Tensor, Parameter]]:
        r"""Return an iterable of the ParameterDict key/value pairs.
        """
        return self._parameters.items()

    def values(self) -> Iterable['Parameter']:
        r"""Return an iterable of the ParameterDict values.
        """
        return self._parameters.values()

    def update(self, parameters: Mapping[str, 'Parameter']) -> None:
        if isinstance(parameters, (OrderedDict, ParameterDict)):
            for key, parameter in parameters.items():
                self[key] = parameter
        elif isinstance(parameters, Mapping):
            for key, parameter in sorted(parameters.items()):
                self[key] = parameter
        else:
            for key, param in parameters:
                self[key] = param
