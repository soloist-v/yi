import yi
import numpy as np
import cv2
from yi import Module, Tensor, tensor, silu, leakyrelu, relu
from yi.nn import (Conv2d, BatchNorm2d, Flatten, Softmax,
                   CrossEntropyLoss, MaxPool2d, Linear, BatchNorm1d, AvgPool2d)
import os


def load_data(data_dir):
    num = 10
    imgs = []
    labels = []
    for parent, _, names in os.walk(data_dir):
        for name in names:
            basename, ext = os.path.splitext(name)
            if ext not in [".jpg", ]:
                continue
            if not basename.isdigit():
                continue
            filepath = os.path.join(parent, name)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)[..., None]
            imgs.append(img)
            labels.append([1 if int(basename) == i else 0 for i in range(num)])
    return np.array(imgs).transpose((0, 3, 1, 2)), np.array(labels)


class LeNet(Module):
    def __init__(self):
        """#input_size=(1*28*28)"""
        super().__init__()
        self.net = yi.Sequential(
            Conv2d(1, 6, 5, padding=2),
            relu,
            MaxPool2d(2, 2),
            # BatchNorm2d(6),
            Conv2d(6, 16, 5),
            relu,
            MaxPool2d(2, 2),
            # BatchNorm2d(16),
            Flatten(),
            Linear(16 * 5 * 5, 120),
            relu,
            BatchNorm1d(120),
            Linear(120, 84),
            relu,
            BatchNorm1d(84),
            Linear(84, 10),
            Softmax(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LimitedList(list):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def append(self, __object) -> None:
        if len(self) >= self.max_size:
            self.pop(0)
        super().append(__object)


if __name__ == '__main__':
    imgs, labels = load_data("../imgs")
    print(imgs.shape, labels.shape)
    model = LeNet()
    lr = 0.001

    data = tensor(imgs)
    labels = tensor(labels)
    print(data.shape)
    loss_func = CrossEntropyLoss()

    # data = model.state_dict()
    # data['net.0.weight'][0] = 5.
    # # print(id(model.net.weight[0]))
    # print(id(getattr(model.net, '0').weight[0]))
    # # print(getattr(model.net, "0"))
    # model.load_state_dict(data)
    # print(id(getattr(model.net, '0').weight[0]))
    # print(getattr(model.net, '0').weight[0])
    # exit()

    optimzer = yi.optim.SGD(model.parameters(), lr=lr, momentum=0.843)
    for i in range(300):
        data_batch = []
        label_batch = []
        for j in np.random.choice(range(len(imgs)), 4):
            data_batch.append(imgs[j])
            label_batch.append(labels[j])
        data_batch = tensor(data_batch)
        label_batch = tensor(label_batch)
        # print(data_batch.shape)
        y = model(data_batch)
        # print(y)
        loss = loss_func(y, label_batch)
        loss.backward()
        # model.step(lr)
        # model.zero_grad()
        optimzer.step()
        optimzer.zero_grad()
        print(loss.data, lr, i)

    model.eval()
    for img in imgs:
        pred = model(tensor(img[None]))[0]
        img = img.transpose(1, 2, 0)
        print(pred.data, np.argmax(pred.data))
        cv2.imshow("src", img)
        cv2.waitKey()
