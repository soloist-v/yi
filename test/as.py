import numpy as np


class A:
    def __init__(self):
        self.dtype = float
        self.shape = (2, 3)
        self.size = 2 * 3

    def __len__(self):
        print("__len__")
        return self.size

    def __getitem__(self, item):
        print("__getitem__")
        return 1

    # @property
    # def dtype(self):
    #     print("dtype")
    #     return float

    # @property
    # def shape(self):
    #     print("shape")
    #     return 2, 3

    @property
    def data(self):
        print("data")
        return [1, 2, 3]


# a = A()
# b = np.zeros_like(a)
# print(b)
print(None or float)