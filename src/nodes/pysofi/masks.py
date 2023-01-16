import numpy as np


def gauss2d_mask(shape=(3, 3), sigma=0.5):
    """
    Generate a 2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma]).
    """
    mdim, ndim = [(pixel-1) / 2 for pixel in shape]
    y, x = np.ogrid[-mdim:(mdim + 1), -ndim:(ndim + 1)]
    h = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gauss1d_mask(shape=(1, 3), sigma=0.5):
    """Generate a 1D gaussian mask."""
    return gauss2d_mask(shape, sigma)[0]