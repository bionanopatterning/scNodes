from numpy.fft import (fftshift, ifftshift, fftn, ifftn, rfftn, irfftn)
import numpy as np
import sys


def _prep_img_and_psf(image, psf):
    """
    From https://github.com/david-hoffman/pyDecon.
    Do basic data checking, convert data to float, normalize psf and make sure
    data are positive.
    """
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    # need to make sure both image and PSF are totally positive.
    image = _ensure_positive(image)
    psf = _ensure_positive(psf)
    # normalize the kernel
    psf /= psf.sum()
    return image, psf


def _ensure_positive(data):
    """
    From https://github.com/david-hoffman/pyDecon.
    Make sure data is positive and has no zeros for numerical stability.
    """
    data = data.copy()
    data[data <= 0] = np.finfo(data.dtype).eps
    return data


def _zero2eps(data):
    """Make sure data is positive and has no zeros."""
    return np.fmax(data, np.finfo(data.dtype).eps)


def zero_pad(image, shape, position='center'):
    """
    From https: // github.com/aboucaud/pypher/blob/master/pypher/pypher.py.
    Extends image to a certain size with zeros.

    Parameters
    ----------
    image : real 2d ndarray
        Input image.
    shape : tuple of list(int)
        Desired output shape of the image.
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner(default)
            * 'center'
                centered

    Returns
    -------
    padded_img: real ndarray
        The zero-padded image.
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img


def psf2otf(psf, shape):
    """
    From https: // github.com/aboucaud/pypher/blob/master/pypher/pypher.py.
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform(FFT) of the point-spread
    function(PSF) array and creates the optical transfer function(OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array(down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up ( or to the left) until the central pixel reaches (1, 1)
    position.
    Adapted from MATLAB psf2otf function.

    Parameters
    ----------
    psf : ndarray
        PSF array.
    shape : list(int)
        Output shape of the OTF array.

    Returns
    -------
    otf : ndarray
        OTF array.
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf


def otf2psf(otf, shape):
    """
    Convert optical transfer function (OTF) to point-spread function (PSF).
    Compute the Inverse Fast Fourier Transform (ifft) of the OTF array and
    creates the PSF array that is not influenced by the OTF off-centering.
    By default, the PSF array is the same size as the OTF array.
    Adapted from MATLAB otf2psf function.

    Parameters
    ----------
    otf : ndarray
        OTF array.
    shape : list (int)
        Output shape of the OTF array.

    Returns
    -------
    psf : ndarray
        PSF array.
    """
    if np.all(otf == 0):
        return np.zeros_like(otf)

    inshape = otf.shape

    # Compute the PSF
    psf = np.fft.ifft2(otf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error.
    n_ops = np.sum(otf.size * np.log2(otf.shape))
    psf = np.real_if_close(psf, tol=n_ops)
    psf = np.real(psf)

    # Circularly shift PSF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(shape):
        psf = np.roll(psf, int(axis_size / 2), axis=axis)

    # Crop output array
    psf = psf[0:shape[0], 0:shape[1]]

    return psf


def corelucy(image, h):
    """
    Make core for the LR estimation. Calculates f to produce the next
    iteration array that maximizes the likelihood that the entire suite
    satisfies the Poisson statistics.
    This is a simplified version of MATLAB corelucy function without
    damping, weights and externally defined functions.

    Parameters
    ----------
    image : ndarray
        Input image.
    h : ndarray
        Zero-padded OTF. h should have the same dimensions as image.

    Returns
    -------
    f : ndarray
        LR extimation core.

    References
    ----------
    .. [1] Acceleration of iterative image restoration algorithms, by D.S.C. Biggs
    and M. Andrews, Applied Optics, Vol. 36, No. 8, 1997.
    .. [2] Deconvolutions of Hubble Space Telescope Images and Spectra,
    R.J. Hanisch, R.L. White, and R.L. Gilliland. in "Deconvolution of Images
    and Spectra", Ed. P.A. Jansson, 2nd ed., Academic Press, CA, 1997.
    """
    u_t = image
    reblur = np.real(ifftn(h * fftn(u_t, u_t.shape), u_t.shape))
    reblur = _ensure_positive(reblur)
    im_ratio = image / reblur
    f = fftn(im_ratio)
    return f


def richardson_lucy(image, psf, iterations=10, **kwargs):
    """
    Richardson-Lucy deconvolution. It deconvolves image using maximum
    likelihood algorithm, returning both deblurred image J and a restored
    point-spread function PSF.
    This is a simplified version of MATLAB deconvblind function without
    damping, weights and externally defined functions.

    Parameters
    ----------
    image : ndarray
       Input degraded image.
    psf : ndarray
       The point spread function.
    iterations : int
       Number of iterations.

    Returns
    -------
    P : ndarray
       Restored point-spread function PSF.
    J : ndarray
        Deblurred image.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.ndimage.gaussian_filter
    >>> image = np.zeros((8,8),dtype=int)
    >>> image[1::2,::2] = 1
    >>> image[::2,1::2] = 1
    >>> psf = np.zeros((7, 7))
    >>> psf[3, 3] = 1
    >>> psf = gaussian_filter(psf, sigma=[7,7])
    >>> new_psf, deconv_im = richardson_lucy(image, psf, 15)

    Notes
    -----
    The quality of the deconvolution result is greatly dependent on the initial
    PSF size instead of the value. We recommend to calibrate PSF of the imaging
    system and use that as the initial PSF guess. Otherwise, generating a PSF
    according to the magnification of the imaging system is an option.

    References
    ----------
    .. [1] http: // en.wikipedia.org/wiki/Richardson % E2 % 80 % 93Lucy_deconvolution
    .. [2] Biggs, D. S. C.; Andrews, M. Acceleration of Iterative Image
    Restoration Algorithms. Applied Optics 1997, 36 (8), 1766.
    """
    # 1. prepare parameters
    image, psf = _prep_img_and_psf(image, psf)
    sizeI, sizePSF = image.shape, psf.shape
    J, P = {}, {}
    J[1], J[2], J[3], J[4] = image, image, 0, np.zeros((np.prod(sizeI), 2))
    P[1], P[2], P[3], P[4] = psf, psf, 0, np.zeros((np.prod(sizePSF), 2))
    WEIGHT = np.ones(image.shape)
    fw = fftn(WEIGHT)

    # 2. L_R iterations
    for k in range(iterations):
        # 2a. make image and PSF predictions for the next iteration
        Y = np.maximum(J[2], 0)
        B = np.maximum(P[2], 0)
        B /= B.sum()
        # 2b. make core for the LR estimation
        H = psf2otf(B, sizeI)
        CC = corelucy(Y, H)
        # 2c. Determine next iteration image & apply positivity constraint
        J[3] = J[2]
        scale = np.real(ifftn(np.multiply(np.conj(H), fw))) + \
                np.sqrt(np.finfo(H.dtype).eps)
        J[2] = np.maximum(np.multiply(image,
                                      np.real(ifftn(np.multiply(np.conj(H), CC)))) / scale, 0)
        J[4] = np.vstack([J[2].T.reshape(-1, ) - Y.T.reshape(-1, ), J[4][:, 1]]).T
        # 2d. Determine next iteration PSF & 
        #     apply positivity constraint + normalization
        P[3] = P[2]
        H = fftn(J[3])
        scale = otf2psf(np.multiply(np.conj(H), fw), sizePSF) + \
                np.sqrt(np.finfo(H.dtype).eps)
        P[2] = np.maximum(np.multiply(B,
                                      otf2psf(np.multiply(np.conj(H), CC), sizePSF)) / scale, 0)
        P[2] /= P[2].sum()
        P[4] = np.vstack([P[2].T.reshape(-1, ) - B.T.reshape(-1, ), P[4][:, 1]]).T
    P, J = P[2], J[2]  # PSF and updated image
    return P, J


def deconvsk(est_psf, input_im, deconv_lambda, deconv_iter):
    """
    Perform serial Richardson-Lucy deconvolution with shrinking PSFs. 
    U = (U**(l/(l-1))) * (U**(l**2/(l-1))) * ... * (U**(l**n/(l-1))).
    The PSF of the imaging system U can be decomposed into a series a 
    smaller (shrinking) PSF U**r where r > 1, and the image can be 
    deconvolved by these PSFs in sequence. 
    In this way, the result is more similar to the input image, so each 
    individual deconvolution step is a lighter deconcolution task.

    Parameters
    ----------
    est_psf : ndarray
        Estimated PSF.
    input_im : ndarray
        Input image that need deconvolution.
    deconv_lambda : float
        Lambda for the exponent between. It is an empirical parameter
        within the range of (1,2).
    deconv_iter : int
        Number of iterations for each deconvolution.

    Returns
    -------
    deconv_im : ndarray
        Deconvoluted image.

    Notes
    -----
    The quality of the deconvolution result is greatly dependent on the initial 
    PSF size instead of the value. We recommend to calibrate PSF of the imaging
    system and use that as the initial PSF guess. Otherwise, generating a PSF 
    according to the magnification of the imaging system is an option. For more
    details on the shrinking kernel deconvolution method, please refer to [1].

    References
    ----------
    .. [1] Xiyu Yi, Sungho Son, Ryoko Ando, Atsushi Miyawaki, and Shimon Weiss, 
    "Moments reconstruction and local dynamic range compression of high order 
    superresolution optical fluctuation imaging," Biomed. Opt. Express 10, 
    2430-2445 (2019).
    """
    xdim, ydim = np.shape(input_im)
    deconv_im = np.append(np.append(input_im, np.fliplr(input_im), axis=1),
                          np.append(np.flipud(input_im), np.rot90(input_im, 2), axis=1), axis=0)
    # Perform mirror extension to the image in order sto surpress ringing
    # artifacts associated with fourier transform due to truncation effect.
    psf0 = est_psf / np.max(est_psf)
    for iter_num in range(deconv_iter):
        alpha = deconv_lambda ** (iter_num + 1) / (deconv_lambda - 1)
        deconv_psf, deconv_im = richardson_lucy(deconv_im, psf0 ** alpha, 1)
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * iter_num,
                                           100 / (deconv_iter - 1) * iter_num))
        sys.stdout.flush()

    deconv_im = deconv_im[0:xdim, 0:ydim]
    return deconv_im