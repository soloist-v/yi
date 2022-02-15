import math
import time
import gc
import numpy as np
import yi
from yi import nn, relu, tensor, Tensor
from yi.nn import Linear, BatchNorm1d
import matplotlib.pyplot as plt


class LimitedList(list):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def append(self, __object) -> None:
        if len(self) >= self.max_size:
            self.pop(0)
        super().append(__object)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(1, 16)
        self.bn1 = BatchNorm1d(16)
        self.l2 = Linear(16, 16)
        self.bn2 = BatchNorm1d(16)
        self.l3 = Linear(16, 1)
        # self.act = yi.MemoryEfficientSwish()
        self.act = yi.MemoryEfficientMish()
        # self.act = yi.AconC(16)
        # self.act = yi.PReLU()
        # self.act = yi.relu
        # self.act = yi.silu

    def zero_grad(self):
        for l in [self.l1, self.l2, self.l3]:
            l.zero_grad()

    def step(self, lr):
        for l in [self.l1, self.l2, self.l3]:
            l.step(lr)

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.bn1(x)
        x = self.l2(x)
        x = self.act(x)
        x = self.bn2(x)
        x = self.l3(x)
        return x


def loss(yp: Tensor, y: Tensor):
    l = 0.5 * ((y - yp) ** 2)
    # return l.mean() * 100
    return yi.sum(l)


if __name__ == '__main__':
    # import yappi

    model = MLP()
    dataset_x = np.arange(0., 10., 0.1)
    dataset_y = np.sin(dataset_x)
    dataset_x = tensor(np.expand_dims(dataset_x, 1))
    dataset_y = tensor(np.expand_dims(dataset_y, 1))
    lr = 0.001
    last_lr = lr
    last_loss = LimitedList(10)
    last_av = np.inf
    count = 0
    isnan = False
    # yappi.clear_stats()
    # yappi.start()
    for i in range(0xfff):
        # gc.enable()
        # gc.collect()
        # gc.disable()
        yp = model(dataset_x)
        res = loss(yp, dataset_y)
        last_loss.append(res.data)
        res.backward()  # 反向传播求梯度
        model.step(lr)  # 根据梯度更新参数
        model.zero_grad()  # 梯度清零
        if i % 100 == 0:
            cur_av = np.average(last_loss)
            if np.isnan(cur_av):
                model.random_parameters()
                lr *= 0.1
            else:
                if np.isnan(last_av):
                    print("add")
                    lr *= 5
            print(cur_av, lr)
            if last_av - cur_av < lr * 0.1:
                count += 1
            if count > 7:
                lr *= 0.5
                count = 0
            last_av = cur_av
            last_lr = lr
    # yappi.stop()
    # stats = yappi.convert2pstats(yappi.get_func_stats())
    # stats.sort_stats("cumulative")
    # stats.print_stats()
    # exit(0)
    plt.plot(dataset_x.data, dataset_y.data, label='real')
    model.eval()
    pred = model(dataset_x)
    # print(pred)
    plt.plot(dataset_x.data, pred.data, label='pred')
    plt.legend()
    plt.show()
