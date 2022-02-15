from .core.tools import *
from .core._tensor import Tensor
import numpy as np


def var(x: Tensor, dim=None):
    """
    方差计算
    :param dim:
    :param x:
    :return:
    """
    print(x)
    av = x.mean(dim=dim)
    # return av

    if dim is None:
        n = int(np.prod(x.shape))
    else:
        n = x.shape[dim]
    print("av", av)
    v = Sub(x, av) ** 2
    # return v
    # print("av", v, n)
    return Sum(v) / n


def std_deviation(x: Tensor, dim=None):
    """
    标准差计算
    :return:
    """
    return sqrt(var(x, dim))
