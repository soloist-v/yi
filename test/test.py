import numpy as np
import yi
from yi import nn, relu, tensor, Tensor
from yi.nn import Linear
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
        self.l2 = Linear(16, 16)
        self.l3 = Linear(16, 1)
        self.act = yi.silu

    def zero_grad(self):
        for l in [self.l1, self.l2, self.l3]:
            l.zero_grad()

    def step(self, lr):
        for l in [self.l1, self.l2, self.l3]:
            l.step(lr)

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        x = self.act(x)
        x = self.l3(x)
        return x


def loss(yp: Tensor, y: Tensor):
    l = 0.5 * ((y - yp) ** 2)
    # print(l.shape)
    # return l.mean() * 100
    return yi.sum(l)


if __name__ == '__main__':
    model = MLP()
    dataset_x = np.arange(0., 10., 0.1).astype(np.float16)
    dataset_y = np.sin(dataset_x)
    dataset_x = tensor(np.expand_dims(dataset_x, 1))
    dataset_y = tensor(np.expand_dims(dataset_y, 1))
    lr = 0.001
    last_loss = LimitedList(10)
    last_av = np.inf
    count = 0
    for i in range(0xfff):
        yp = model(dataset_x)
        res = loss(yp, dataset_y)
        last_loss.append(res.data)
        # print("loss:", res, lr)
        res.backward()  # 反向传播求梯度
        model.step(lr)  # 根据梯度更新参数
        model.zero_grad()  # 梯度清零
        if i % 100 == 0:
            cur_av = np.average(last_loss)
            print(cur_av, lr)
            if last_av - cur_av < lr * 0.1:
                count += 1
            if count > 20:
                lr *= 0.5
                count = 0
            last_av = cur_av
    plt.plot(dataset_x.data, dataset_y.data, label='real')
    pred = model(dataset_x)
    # print(pred)
    plt.plot(dataset_x.data, pred.data, label='pred')
    plt.legend()
    plt.show()
