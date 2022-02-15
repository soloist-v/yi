import numpy as np
from .tools import tensor


def arange(start, *args, **kwargs):
    data = np.arange(start, *args, **kwargs)
    return tensor(data, data.dtype)
