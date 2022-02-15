from .base import Module
from ..core import Union, Tuple, Im2Col, np


class AvgPool2d(Module):

    def __init__(self, ksize: Union[int, Tuple[int, int]], stride=1, padding=0):
        super().__init__()
        self.h_filter, self.w_filter = (ksize, ksize) if isinstance(ksize, int) else ksize
        self.s = stride
        self.p = padding

    def forward(self, x):
        m, n_c_prev, n_h_prev, n_w_prev = x.shape
        n_c = n_c_prev
        n_h = int((n_h_prev + 2 * self.p - self.h_filter) / self.s) + 1
        n_w = int((n_w_prev + 2 * self.p - self.w_filter) / self.s) + 1
        x_col = Im2Col(x, self.h_filter, self.w_filter, self.s, self.p)
        x_col = x_col.reshape(n_c, x_col.shape[0] // n_c, -1)
        a_pool = x_col.mean(dim=1)
        # ---------------------
        out = a_pool.reshape(n_c_prev, n_h, n_w, m)
        out = out.transpose(3, 0, 1, 2)
        # ---------------------
        return out


class MaxPool2d(Module):

    def __init__(self, size, stride=1):
        super().__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        n_x, d_x, h_x, w_x = x.shape
        h_out = (h_x - self.size) / self.stride + 1
        w_out = (w_x - self.size) / self.stride + 1
        h_out, w_out = int(h_out), int(w_out)
        # ------------------------------------------------------------------------------
        x_reshaped = x.reshape(
            x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])
        x_col = Im2Col(x_reshaped, self.size, self.size, padding=0, stride=self.stride)
        max_indexes = np.argmax(x_col.data, axis=0)
        out = x_col[max_indexes, range(max_indexes.size)]
        out = out.reshape(h_out, w_out, n_x, d_x).transpose(2, 3, 0, 1)
        return out
