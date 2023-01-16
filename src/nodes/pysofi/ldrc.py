import numpy as np
from scipy.interpolate import griddata
import sys


def ldrc(mask_im, input_im, order=1, window_size=[25, 25]):
    """
    Process the image array with "local dynamic range compression" (ldrc).

    Parameters
    ----------
    mask_im : ndarray
        A reference image.
        Usually a average/sum image or second-order SOFI image is used.
    input_im : ndarray
        An input image, usually a high-order moment- or cumulant-
        reconstructed image.
    order : int
        The order of the reconstructed image.
    window_size : [int, int]
        The [x, y] dimension of the scanning window.

    Returns
    -------
    ldrc_im : ndarray
        The compressed image with the same dimensions of input_im.

    Notes
    -----
    High-order cumulants or moments reconstructions result-in images with a
    large dynamic range of pixel intensities. This ldrc algorithm compresses
    the dynamic range of these reconstructions with respect to a reference
    image while retaining resolution enhancement.
    The compression is performed locally in a small window that is scanned
    across the image. For details of the ldrc method, see [1].

    References
    ----------
    .. [1] Xiyu Yi, Sungho Son, Ryoko Ando, Atsushi Miyawaki, and Shimon Weiss,
    "Moments reconstruction and local dynamic range compression of high order
    superresolution optical fluctuation imaging," Biomed. Opt. Express 10,
    2430-2445 (2019).
    """
    xdim_mask, ydim_mask = np.shape(mask_im)
    xdim, ydim = np.shape(input_im)
    if xdim == xdim_mask and ydim == ydim_mask:
        mask = mask_im
    else:
        # Resize mask to the image dimension if not the same dimension
        mod_xdim = (xdim_mask-1)*order + 1    # new mask x dimemsion
        mod_ydim = (ydim_mask-1)*order + 1    # new mask y dimemsion
        px = np.arange(0, mod_xdim, order)
        py = np.arange(0, mod_ydim, order)

        # Create coordinate list for interpolation
        coor_lst = []
        for i in px:
            for j in py:
                coor_lst.append([i, j])
        coor_lst = np.array(coor_lst)

        orderjx = complex(str(mod_xdim) + 'j')
        orderjy = complex(str(mod_ydim) + 'j')
        # New coordinates for interpolated mask
        px_new, py_new = np.mgrid[0:mod_xdim-1:orderjx, 0:mod_ydim-1:orderjy]

        interp_mask = griddata(coor_lst, mask_im.reshape(-1, 1),
                               (px_new, py_new), method='cubic')
        mask = interp_mask.reshape(px_new.shape)

    seq_map = np.zeros((xdim, ydim))
    ldrc_im = np.zeros((xdim, ydim))
    for i in range(xdim - window_size[0] + 1):
        for j in range(ydim - window_size[1] + 1):
            window = input_im[i:i+window_size[0], j:j+window_size[1]]
            norm_window = (window - np.min(window)) / \
                          (np.max(window) - np.min(window))
            # norm_window = window / np.max(window)
            ldrc_im[i:i+window_size[0], j:j+window_size[1]] = \
                ldrc_im[i:i+window_size[0], j:j+window_size[1]] + \
                norm_window * \
                np.max(mask[i:i+window_size[0], j:j+window_size[1]])
            seq_map[i:i+window_size[0], j:j+window_size[1]] = \
                seq_map[i:i+window_size[0], j:j+window_size[1]] + 1
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" %
                         ('='*int(20*(i+1)/(xdim - window_size[0] + 1)),
                          100*(i+1)/(xdim - window_size[0] + 1)))
        sys.stdout.flush()
    ldrc_im = ldrc_im / seq_map
    return ldrc_im
