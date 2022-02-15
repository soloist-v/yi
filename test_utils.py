import time

import numpy as np
from typing import *
import yi
import time
import torch

if __name__ == '__main__':
    a = yi.tensor([1, 2, 3.])
    b = yi.tensor(1., requires_grad=True)
    y = a * b
    y = y.sum()
    y.backward()
    print(b.grad)
    exit()
    # a = torch.tensor([1, 2, 2.], requires_grad=True)
    # print(a.dtype)
    # exit()
    # np.NPY_DEFAULT_TYPE = np.float32
    data = np.array([[[1., 2, 10], [3, 6, 25]]], np.float32)
    # a = yi.tensor(data, requires_grad=True)  # 0.16666667
    a = torch.tensor(data, requires_grad=True)
    dim = (2, 1)
    # res = a.mean(dim=(2, 1), keep_dim=True)
    res = torch.max(a, dim=0, keepdim=True)
    print(res)
    # torch.max()
    # res = res.sum()
    res.backward()
    print(res)
    print(a.grad)
    exit()
    b = np.array([[[1., 2], [3, 5]]], np.float32)
    # c = np.emath.logn(a, b)
    # c = a * b
    c = np.maximum(a, 0.)
    print(c.dtype)
    exit()
    t0 = time.time()
    # print(np.array(a))
    print(time.time() - t0)
    mask = np.random.binomial(1, 0.5, size=a.shape) / 0.5
    print(mask.dtype)
    print(a.dtype == np.float)
    # a = np.random.rand(10000, 10000)
    # # b = np.random.rand(10000, 10000)
    # t0 = time.time()
    # for i in range(10):
    #     y = np.array(a, copy=False, dtype="float32")
    # print(time.time() - t0)
    # a = torch.tensor([1, 2, 3], dtype=torch.float32)
    # print(a.dtype)
    # a = yi.tensor([[1, 2], [3, 4]], requires_grad=True)
    # a = yi.tensor([[[1, 2.], [5, 8]], [[3, 4], [7, 5]]], requires_grad=True)
    # b = a * a
    # b = b.mean(0, keepdim=True)
    # y = b.sum()
    # y.backward()
    # print(a.grad)
    # a = yi.to_tensor(np.ones((2, 3, 16, 16)), requires_grad=True)
    # conv2d = yi.Conv2D(3, 64, 3, 1, 1)
    # # av_pool = yi.MaxPool(3)
    # # print(a)
    # b = conv2d(a)
    # # b = av_pool(a)
    # print("b shape", b.shape)
    # b.backward()
    # print(conv2d.w.grad)
    # t0 = time.time()
    # for i in range(9999):
    #     yi.as_tensor(i)
    # print(time.time() - t0)
    # time.sleep(5)
