# yi

Yi is a simple deep learning framework based on Numpy that supports automatic gradients.
## install
`python setup.py install`

## Example:

#### 1.Define network

```python
from yi import Tensor
from yi.nn import (Conv2d, Flatten, Softmax, Module, Sequential,
                   MaxPool2d, Linear, BatchNorm1d, ReLU)


class LeNet(Module):
    def __init__(self):
        """#input_size=(1*28*28)"""
        super().__init__()
        self.net = Sequential(
            Conv2d(1, 6, 5, padding=2),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(6, 16, 5),
            ReLU(),
            MaxPool2d(2, 2),
            Flatten(),
            Linear(16 * 5 * 5, 120),
            ReLU(),
            BatchNorm1d(120),
            Linear(120, 84),
            ReLU(),
            BatchNorm1d(84),
            Linear(84, 10),
            Softmax(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

```

#### 2.Define activation function, No need to implement backward, autograd

```python
from yi import Exp, Log, Tensor
import numpy as np


def sigmoid(x):
    return 1 / (1 + Exp(-x))


def tanh(x: Tensor):
    E = np.e
    return (E ** x - E ** (-x)) / (E ** x + E ** (-x))


def softplus(x: Tensor) -> Tensor:
    return Log.apply(1 + Exp.apply(x))


def mish(x: Tensor):
    return x * tanh(softplus(x))


def silu(x: Tensor):
    return x * sigmoid(x)

```

#### 3.Custom operator

```python
from yi import Function, sigmoid, tanh, softplus


class MemoryEfficientSwish(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = sigmoid(x)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MemoryEfficientMish(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * (tanh(softplus(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = sigmoid(x)
        fx = tanh(softplus(x))
        return grad_output * (fx + x * sx * (1 - fx * fx))


# or
from yi import (ComputeNodeDetached, ndarray, np, Union,
                apply_broadcast_a, apply_broadcast_b)


class Maximum(ComputeNodeDetached):
    def __init__(self, a, b):
        super().__init__(a, b)

    def forward(self) -> ndarray:
        a, b = self.inputs_data
        return np.maximum(a, b)
    
    # backward + (a,b,c,d,e,f,g...)
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
```

#### 4.Higher derivative

```python
import yi

a = yi.tensor([1, 2, 3, 4, 5, 6.], requires_grad=True)
y = a * a * a * 2 + 3 * a * a + a
print("y", y)
grad1 = yi.grad(y, a, retain_graph=True)
print("grad1", grad1)
grad2 = yi.grad(grad1, a, retain_graph=True)
print("grad2", grad2)
grad3 = yi.grad(grad2, a, retain_graph=True)
print("grad3", grad3)
grad4 = yi.grad(grad3, a, retain_graph=True)
print("grad4", grad4)
# out:
# y Tensor([  6.  30.  84. 180. 330. 546.], requires_grad=True, shape=(6,), dtype=float32, grad_fn=<yi.core.nodes.Add object at 0x0000024BACD2DF40>)
# grad1 Tensor([ 13.  37.  73. 121. 181. 253.], requires_grad=True, shape=(6,), dtype=float32, grad_fn=<yi.core.nodes.Add object at 0x0000024BACD41040>)
# grad2 Tensor([18. 30. 42. 54. 66. 78.], requires_grad=True, shape=(6,), dtype=float32, grad_fn=<yi.core.nodes.Add object at 0x0000024BACD4DE50>)
# grad3 Tensor([12. 12. 12. 12. 12. 12.], requires_grad=True, shape=(6,), dtype=float32, grad_fn=<yi.core.nodes.Add object at 0x0000024BACD59D30>)
# grad4 0.0
```
