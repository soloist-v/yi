# import torch
# import torch.nn as nn
import numpy as np
import torch
from torch.optim import SGD


class AAA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, 3, padding=1),
            torch.nn.Conv2d(1, 1, 3, padding=1),
            torch.nn.Conv2d(1, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class BBB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = AAA()

    def forward(self, x):
        return self.a(x)


class CCC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = BBB()

    def forward(self, x):
        return self.b(x)


model = AAA()
model()
exit()
print(id(model.conv.weight[0]))
data = model.state_dict()
print(type(data))
print(getattr(model.net, '0'))
print(list(data.keys()))
# data["conv.weight"] = model.conv.weight.clone()
model.load_state_dict(data)
# print(id(model.conv.weight[0]))
exit()
print(hash(model), hash(id(model)))
model.__hash__()
a = torch.tensor([1, 2, 3])
print(hash(a.__hash__()), id(a))
o = object()
print(hash(o), id(o))
for name, v in model.state_dict().items():
    print(name)
# for p in model.parameters():
#     print(p)
torch.optim.SGD
