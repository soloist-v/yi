from ..core import Tensor, float32, tensor


class Parameter(Tensor):
    def __init__(self, data, dtype=float32, requires_grad=True):
        super().__init__(tensor(data, dtype), requires_grad=requires_grad)
