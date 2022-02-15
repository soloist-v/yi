from .base import Module
from ..core import np, Reshape, hsplit, UnSqueeze, concatenate, Im2Col, float32
from .parameter import Parameter
from .functional import im2col


class Conv2d0(Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.need_bias = bias
        w = np.random.randn(self.out_channels, self.in_channels, self.ksize, self.ksize) * np.sqrt(1. / self.ksize)
        self.w = Parameter(w)
        if self.need_bias:
            bias = np.random.randn(self.out_channels) * np.sqrt(1. / self.out_channels)
            self.bias = Parameter(bias)
        self.cache = None

    def forward(self, input_x):
        m, n_c_prev, n_h_prev, n_w_prev = input_x.shape
        n_c = self.out_channels
        n_h = int((n_h_prev + 2 * self.padding - self.ksize) / self.stride) + 1
        n_w = int((n_w_prev + 2 * self.padding - self.ksize) / self.stride) + 1
        x_col = im2col(input_x, self.ksize, self.ksize, self.stride, self.padding)
        w_col = Reshape(self.w, (self.out_channels, -1))
        # Perform matrix multiplication.
        if self.need_bias:
            b_col = Reshape(self.bias, (-1, 1, 1))
            out = w_col @ x_col + b_col
        else:
            out = w_col @ x_col
        # Reshape back matrix to image.
        out = hsplit(out, m)
        res = []
        for i in out:
            a = UnSqueeze(i, 0)
            res.append(a)
        out = concatenate(*res, dim=0).reshape((m, n_c, n_h, n_w))
        return out


class Conv2d(Module):

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h_filter, self.w_filter = (ksize, ksize) if isinstance(ksize, int) else ksize
        self.kernel_size = (self.w_filter, self.h_filter)
        self.stride, self.padding = stride, padding
        self.weight = Parameter(np.random.randn(
            self.out_channels, self.in_channels, self.h_filter, self.w_filter) / np.sqrt(self.out_channels / 2.))
        if bias:
            self.bias = Parameter(np.zeros((self.out_channels, 1), float32))
        else:
            self.bias = None

    def forward(self, input_x):
        # if not is_tensor(X):
        #     X = tensor(X)
        xn, xc, xh, xw = input_x.shape
        h_out = (xh - self.h_filter + 2 * self.padding) / self.stride + 1
        w_out = (xw - self.w_filter + 2 * self.padding) / self.stride + 1
        h_out, w_out = int(h_out), int(w_out)
        # ----------------------------------------------------------------------------
        n_x = input_x.shape[0]
        x_col = Im2Col(input_x, self.h_filter, self.w_filter, stride=self.stride, padding=self.padding)
        w_row = self.weight.reshape(self.out_channels, -1)
        if self.bias is None:
            out = w_row @ x_col
        else:
            out = w_row @ x_col + self.bias
        out = out.reshape(self.out_channels, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        return out
