from .base import Module
from ..core import dropout


class Dropout(Module):

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        return dropout(x, self.prob)
