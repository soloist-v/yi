from .base import Module
from ..core import log, sum


class CrossEntropyLoss(Module):
    """
    用于度量两个概率分布 【一批数据的预测概率分布】和【这批数据的真实概率分布(因为是一组数据所以是这组数据的概率分布)】之间的【差异】
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        loss = -sum(y * log(y_pred))
        return loss
