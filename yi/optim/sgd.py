from typing import Iterable, Union
from .optimizer import Optimizer
from ..nn import Tensor, Parameter, np


class SGD(Optimizer):

    def __init__(self, params: Iterable[Union[Parameter, Tensor]], lr=0.001, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=False):
        super(SGD, self).__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.buffers = {}
        for param in params:
            self.buffers[param] = np.zeros_like(param.data)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        weight_decay = self.weight_decay
        momentum = self.momentum
        dampening = self.dampening
        nesterov = self.nesterov
        for p in self.params:
            if p.grad is None:
                continue
            d_p = p.grad
            if isinstance(d_p, Tensor):
                d_p = d_p.data
            if weight_decay != 0:
                d_p = d_p + p * weight_decay
            if momentum != 0:
                buf = self.buffers[p]
                buf[:] = buf * momentum + d_p * (1 - dampening)
                if nesterov:
                    d_p = d_p + buf * momentum
                else:
                    d_p = buf
            p.add_(d_p, alpha=-self.lr)
        return loss


class SGD1(Optimizer):
    def __init__(self, params: Iterable[Union[Parameter, Tensor]], lr=0.001, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}
        for param in params:
            self.velocities[param] = np.zeros_like(param.data)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for p in self.params:
            if p.grad is None:
                continue
            d_p = p.grad
            if isinstance(d_p, Tensor):
                d_p = d_p.data
            self.velocities[p] = self.momentum * self.velocities[p] - self.lr * d_p
            p.add_(self.velocities[p])
        return loss
