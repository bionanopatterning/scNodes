"""
This module performs fourier interpolation on a single image 
or a image stack (tiff file). Interpolated image (stack) can be 
saved into a new tiff file or return as a numpy array.

Fourier interpolation was implemented with transformation matrix 
operation, where the Fourier transformation matrix was constructed 
with the original matrix size without interpolation grids, and the 
inverse Fourier transformation matrix encodes the extra interpolation
position coordinates.

Adapted from https://github.com/xiyuyi/xy_fInterp_forTIFF.
"""

import numpy as np

try:
    from . import switches as s
except:
    import switches as s

if s.SPHINX_SWITCH is False:
    import tifffile as tiff

import sys


def _base_vect_generator2d(xrange, yrange):
    """Generate base vectors for x- and y-dimension."""
    bx, by = np.zeros(xrange), np.zeros(yrange)
    bx[1], by[1] = 1, 1
    bx, by = np.fft.fft(bx), np.fft.fft(by)
    # bx, by = fftshift(fft(ifftshift(bx))), fftshift(fft(ifftshift(by)))
    return bx, by


def _calc_ft_matrix(base, spectrum_range):
    """
    Calculate fourier transform matrix without center shift from base vectors.
    """
    power_matrix = np.ones((spectrum_range, spectrum_range))
    power_matrix = np.arange(spectrum_range).reshape(spectrum_range, 1) * \
                   power_matrix
    ft_matrix = np.power(base, power_matrix)
    return ft_matrix


def _calc_ift_matrix(base, spectrum_range, interp_num):
    """
    Calculate inverse fourier transform matrix without center shift from base
    vectors for a given fold of interpolation.
    """
    conj_base = np.reshape(np.conj(base), (1, spectrum_range))
    ift = np.matmul(np.ones(((spectrum_range - 1) * interp_num + 1, 1)),
                    conj_base)
    iftp = np.arange(0, spectrum_range - 1 + 1e-10, 1 / interp_num)
    iftp = np.matmul(iftp.reshape(-1, 1), np.ones((1, spectrum_range)))
    ift = np.power(ift, iftp) / spectrum_range
    return ift


def ft_matrix2d(xrange, yrange):
    """Calculate fourier transform matrix for x- and y-dimension."""
    bx, by = _base_vect_generator2d(xrange, yrange)
    fx = _calc_ft_matrix(bx, xrange)
    fy = _calc_ft_matrix(by, yrange).T
    return fx, fy


def ift_matrix2d(xrange, yrange, interp_num):
    """
    Calculate inverse fourier transform matrix without center shift for a 
    given fold of interpolation for x- and y-dimension. Here we use 2 times 
    the x-dimension of a frame image for xrange and 2 times of y for yrange. 
    interp_num is the number of times the resolution enhanced. For instance, 
    when interp_num = 2, the resolution is two times the original one.
    """
    bx, by = _base_vect_generator2d(xrange, yrange)
    ifx = _calc_ift_matrix(bx, xrange, interp_num)
    ify = _calc_ift_matrix(by, yrange, interp_num).T

    return ifx, ify


def interpolate_image(im, fx, fy, ifx, ify, interp_num):
    """
    Performs fourier interpolation to increase the resolution of the image. 
    The interpolated image can be further processed by SOFI for 
    super-resolution imaging.

    Parameters
    ----------
    im : ndarray
        Input image.
    fx : ndarray
        Fourier transform matrix in x generated from ft_matrix2d function.
    fy : ndarray
        Fourier transform matrix in y generated from ft_matrix2d function.
    ifx : ndarray
        Inverse fourier transform matrix in x from ift_matrix2d function.
    ify : ndarray
        Inverse fourier transform matrix in y from ift_matrix2d function.
    interp_num : int
        The number of pixels to be interpolated between two adjacent pixels.

    Returns
    -------
    interp_im : ndarray
        Interpolated image with new dimensions.

    Notes
    -----
    Please note here that interpolation doesn' extend over the edgeo f the 
    matrix, therefore the total number of pixels in the resulting matrix is 
    not an integer fold of the original size of the matrix. For example, if 
    the original matrix size of each frame is xdim and ydim for x and y 
    dimensions respectively. After interpolation, the size of the resulting 
    matrix will be ((xdim-1)*f + 1) for x dimension, and ((ydim-1)*f + 1) 
    for y dimension.

    Example
    -------
    ::

        import numpy as np
        import matplotlib.pyplot as plt
        xdim, ydim, sigma, mu = 10, 10, 0.5, 0
        x, y = np.meshgrid(np.linspace(-1, 1, xdim), np.linspace(-1, 1, xdim))
        im = np.exp(-((np.sqrt(x*x + y*y) - mu)**2 / (2*sigma**2)))
        xrange, yrange, interp_num = 2 * xdim, 2 * ydim, 3
        fx, fy = ft_matrix2d(xrange, yrange)
        ifx, ify = ift_matrix2d(xrange, yrange, interp_num)
        interp_im = interpolate_image(im, fx, fy, ifx, ify, interp_num)

    """
    xdim, ydim = np.shape(im)

    # 1. Extend im to create the natural peoriocity in the resulting image 
    # to avoid ringing artifacts after fourier interpolation.
    ext_im = np.append(np.append(im, np.fliplr(im), axis=1),
                       np.append(np.flipud(im), np.rot90(im, 2),
                                 axis=1), axis=0)

    # 2. Fourier transform
    fall = np.matmul(np.matmul(fx, ext_im), fy)
    # fall=fx@ext_im@fy for Python 3.5 or newer versions

    # 3. Inverse Fourier transform
    ifall = np.absolute(np.dot(np.dot(ifx, fall), ify))

    # 4. Take the region corresponding to the FOV of the original image
    xdim_new = (xdim - 1) * interp_num + 1
    ydim_new = (ydim - 1) * interp_num + 1
    interp_im = ifall[:xdim_new, :ydim_new]

    return interp_im


def fourier_interp_array(im, interp_num_lst):
    """
    Performs fourier interpolation on an image array with a list of 
    interpolation factors.
    """
    xdim, ydim = im.shape
    xrange, yrange = 2 * xdim, 2 * ydim  # define the ft spectrum span
    fx, fy = ft_matrix2d(xrange, yrange)
    interp_im_lst = []
    for interp_num in interp_num_lst:
        ifx, ify = ift_matrix2d(xrange, yrange, interp_num)
        interp_im = interpolate_image(im, fx, fy, ifx, ify, interp_num)
        # interp_im_lst.append(np.int_(np.around(interp_im)))
        interp_im_lst.append(interp_im)

    return interp_im_lst


def fourier_interp_tiff(filepath, filename, interp_num_lst, frames=[],
                        save_option=True, return_option=False):
    """
    Performs fourier interpolation on a tiff image (stack) with a list of 
    interpolation factors (the number of pixels to be interpolated between 
    two adjacent pixels).

    Parameters
    ----------
    filepath : str
        The path to the tiff file.
    filename : str
        The filename of the tiff file without '.tif'.
    interp_num_lst : list (int)
        A list of interpolation factors.
    frames : list of int
        The start and end frame number.
    save_option : bool
        Whether to save the interpolated images into tiff files (each 
        interpolation factor seperately).
    return_option : bool
        Whether to return the interpolated image series as 3d arrays.

    Returns
    -------
    interp_imstack_lst : list (ndarray)
        A list of interpolated image series corresponding to the interpolation
        factor list.
    """
    imstack = tiff.TiffFile(filepath + '/' + filename + '.tif')
    xdim, ydim = np.shape(imstack.pages[0])
    xrange, yrange = 2 * xdim, 2 * ydim
    fx, fy = ft_matrix2d(xrange, yrange)
    interp_imstack_lst = []

    # if user did not select the video length, process on the whole video
    if frames:
        for interp_num in interp_num_lst:
            ifx, ify = ift_matrix2d(xrange, yrange, interp_num)
            interp_imstack = []
            print('Calculating interpolation factor = %d...' % interp_num)
            for frame_num in range(frames[0], frames[1]):
                # read a frame from the original tiff file
                im = tiff.imread(filepath + '/' + filename + '.tif', key=frame_num)
                # find the minimum and maximum values of this image
                immax = np.max(im.ravel())
                immin = np.min(im.ravel())
                # perform interpolation
                interp_im = interpolate_image(im, fx, fy, ifx, ify, interp_num)
                # ensure the interpolated image have the identical dynamic range of the original image
                interp_immax = np.max(interp_im.ravel())
                interp_immin = np.min(interp_im.ravel())
                interp_im = (interp_im - interp_immin) / (interp_immax - interp_immin)
                interp_im = interp_im * (immax - immin) + immin

                #                interp_im = np.int_(np.around(interp_im))
                interp_im = np.uint16(np.around(interp_im))
                sys.stdout.write('\r')
                sys.stdout.write("[{:{}}] {:.1f}%".format(
                    "=" * int(30 / (frames[1] - frames[0]) * (frame_num - frames[0] + 1)), 29,
                    (100 / (frames[1] - frames[0]) * (frame_num - frames[0] + 1))))
                sys.stdout.flush()
                if save_option is True:
                    tiff.imwrite(filepath + '/' + filename + '_InterpNum' + str(interp_num) +
                                 '.tif', interp_im, dtype='uint16', append=True)
                if return_option is True:
                    interp_imstack.append(interp_im)

            if return_option is True:
                interp_imstack_lst.append(interp_imstack)
            print('\n')
    else:
        mvlength = len(imstack.pages)
        for interp_num in interp_num_lst:
            ifx, ify = ift_matrix2d(xrange, yrange, interp_num)
            interp_imstack = []
            print('Calculating interpolation factor = %d...' % interp_num)
            for frame_num in range(mvlength):
                # read a frame from the original tiff file
                im = tiff.imread(filepath + '/' + filename + '.tif', key=frame_num)
                # find the minimum and maximum values of this image
                immax = np.max(im.ravel())
                immin = np.min(im.ravel())
                # perform interpolation
                interp_im = interpolate_image(im, fx, fy, ifx, ify, interp_num)
                # ensure the interpolated image have the identical dynamic range of the original image
                interp_immax = np.max(interp_im.ravel())
                interp_immin = np.min(interp_im.ravel())
                interp_im = (interp_im - interp_immin) / (interp_immax - interp_immin)
                interp_im = interp_im * (immax - immin) + immin

                #                interp_im = np.int_(np.around(interp_im))
                interp_im = np.uint16(np.around(interp_im))
                sys.stdout.write('\r')
                sys.stdout.write("[{:{}}] {:.1f}%".format(
                    "=" * int(30 / mvlength * (frame_num + 1)), 29,
                    (100 / mvlength * (frame_num + 1))))
                sys.stdout.flush()
                if save_option is True:
                    tiff.imwrite(filepath + '/' + filename + '_InterpNum' + str(interp_num) +
                                 '.tif', interp_im, dtype='uint16', append=True)
                if return_option is True:
                    interp_imstack.append(interp_im)

            if return_option is True:
                interp_imstack_lst.append(interp_imstack)
            print('\n')

    if return_option is True:
        return interp_imstack_lst