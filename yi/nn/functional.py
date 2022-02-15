from ..core import Tensor, np, pad_constant, concatenate


def get_indices(x_shape, kernel_height, kernel_width, stride, padding):
    """
        Returns index matrices in order to transform our input image into a matrix.
        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.
        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d.
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = x_shape

    # get output size
    out_h = int((n_H + 2 * padding - kernel_height) / stride) + 1
    out_w = int((n_W + 2 * padding - kernel_width) / stride) + 1

    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(kernel_height), kernel_width)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----

    # Slide 1 vector.
    slide1 = np.tile(np.arange(kernel_width), kernel_height)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), kernel_height * kernel_width).reshape(-1, 1)

    return i, j, d


def im2col(x: "Tensor", kernel_height, kernel_width, stride, padding) -> "Tensor":
    """
        Transforms our input image into a matrix.
        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.
        Returns:
        -cols: output matrix.
    """
    # Padding
    x_padded = pad_constant(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    i, j, d = get_indices(x.shape, kernel_height, kernel_width, stride, padding)
    # Multi-dimensional arrays indexing.
    cols = x_padded[:, d, i, j]
    cols = concatenate(cols, dim=-1)
    return cols
