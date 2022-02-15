from ..nn import Conv2d, BatchNorm2d
from ..core import sqrt, dot, zeros, tensor
import numpy as np


def fuse_conv_and_bn(conv: Conv2d, bn: BatchNorm2d):
    bn.eval()
    conv.eval()
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = Conv2d(conv.in_channels,
                       conv.out_channels,
                       ksize=conv.kernel_size,
                       stride=conv.stride,
                       padding=conv.padding,
                       bias=True).requires_grad_(False)
    # prepare filters
    w_conv = conv.weight.copy().view(conv.out_channels, -1)
    w_bn = tensor(np.diag((bn.gamma / (sqrt(bn.eps + bn.moving_var))).data))
    fusedconv.weight.copy_(dot(w_bn, w_conv).view(fusedconv.weight.shape))
    # prepare spatial bias
    b_conv = zeros(conv.weight.size(0)) if conv.bias is None else conv.bias
    b_bn = bn.beta - bn.gamma * bn.moving_mean / sqrt(bn.moving_var + bn.eps)
    fusedconv.bias.copy_(dot(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv
