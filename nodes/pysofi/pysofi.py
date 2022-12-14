from nodes.pysofi import deconvsk, ldrc, masks, reconstruction
from nodes.pysofi import finterp as f
from nodes.pysofi import switches as s

if s.SPHINX_SWITCH is False:
    import tifffile as tiff

import numpy as np


class PysofiData:
    """
    Data object contains the information of a dataset (e.g. dimensions,
    frame numbers), and provides SOFI methods to perform analysis and 
    visualization on the dataset (moments reconstruction, cumulants 
    reconstruction, shrinking kernel deconvolution, etc...). 

    When loading a tiff file, a PysofiData() object is created and further 
    SOFI analysis can be preformed. All the standard data-attributes 
    are listed below. New data-attributes can be added or updated using 
    '.add()' function.

    Parameters
    ----------
    filapath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.

    Attributes
    ----------
    filename : str
    filepath : str
    ave : ndarray
        The average image of the image stack / tiff video.
    finterp_factor : int
        The interpolation factor for Fourier interpolation.
    morder_lst : list
        All orders of moments-reconstructions that have been calculated.
    morder_finterp_lst : list
        All orders of moments-reconstructions after Fourier interpolation
        that have been calculated.
    moments_set : dict
        moment order (int) -> moment-reconstructed image (ndarray)
        A dictionary of orders and corrensponding moment reconstructions.
    moments_set_bc : dict
        moment order (int) -> moment-reconstructed image (ndarray)
        A dictionary of orders and corrensponding moment reconstructions 
        with bleaching correction.
    cumulants_set : dict
        cumulant order (int) -> cumulant-reconstructed image (ndarray)
        A dictionary of orders and corrensponding cumulant reconstructions.  
    cumulants_set_bc : dict 
        cumulant order (int) -> cumulant-reconstructed image (ndarray)
        A dictionary of orders and corrensponding cumulant reconstructions 
        with bleaching correction.   
    morder : int
        The highest order of moment reconstruction that has been calculated.
    corder : int
        The highest order of cumulant reconstruction that has been 
        calculated.  
    fbc : float
        Bleaching correction factor.
    n_frames : int
        The number of frames of the image stack / tiff video.
    xdim : int
        The number of pixels in x dimension.
    ydim : int
        The number of pixels in y dimension.


    Notes
    -----
    For SOFI processing, after loading the tiff video into the Data object, 
    a pipeline of 1) fourier interpolation, 2) moments reconstrtuction or 
    cumulants reconstruction, 3) noise filtering 1, 4) shrinking kernel de-
    convolution, 5) noise filtering 2, and 6) ldrc. The processed video will
    be saved into a new tiff file with the colormap user selects.

    References
    ----------
    .. [1] Xiyu Yi, Sungho Son, Ryoko Ando, Atsushi Miyawaki, and Shimon Weiss, 
    "Moments reconstruction and local dynamic range compression of high order 
    superresolution optical fluctuation imaging," Biomed. Opt. Express 10, 
    2430-2445 (2019).
    """

    def __init__(self, filepath, filename):
        self.filename = filename
        self.filepath = filepath
        self.ave = None
        self.finterp_factor = 1
        self.morder_lst = []
        self.morder_finterp_lst = []
        self.moments_set = {}
        self.moments_set_bc = {}
        self.moments_finterp_set = {}
        self.cumulants_set = {}
        self.cumulants_set_bc = {}
        self.morder = 0
        self.corder = 0
        self.fbc = 1
        self.n_frames, self.xdim, self.ydim = self.get_dims()

    def average_image(self, frames=[]):
        """Calculate average image of the tiff video."""
        self.ave = reconstruction.average_image(self.filepath, self.filename, frames)
        return self.ave

    def average_image_with_finterp(self, interp_num):
        if self.ave is not None:
            finterp_ave = f.fourier_interp_array(self.ave, [interp_num])
            return finterp_ave[0]
        else:
            finterp_ave = reconstruction.average_image_with_finterp(
                self.filepath, self.filename, interp_num)
            return finterp_ave

    def moment_image(self, order=6, mean_im=None, mvlength=[],
                     finterp=False, interp_num=1):
        """
        Calculate the moment-reconstructed image of a defined order.

        Parameters
        ----------
        order : int
            The order number of the moment-reconstructed image.
        mean_im : ndarray
            Average image of the tiff stack.
        mvlength : int
            Length of the video to calculate moments-reconstruction.
        finterp : bool
            Whether to first conduct Fourier interpolation then calculate
            moments-reconstructions.
        interp_num : int
            The interpolation factor.

        Returns
        -------
        moment_im : ndarray
            The calculated moment-reconstructed image.
        """
        if finterp is False:
            if order in self.morder_lst:
                print("this order has been calculated")
                print('\n')
            moment_im = reconstruction.calc_moment_im(self.filepath,
                                                      self.filename,
                                                      order, mvlength)
            self.morder_lst.append(order)
            self.moments_set[order] = moment_im
            return self.moments_set[order]
        else:
            if self.finterp_factor != 1 and interp_num != self.finterp_factor:
                print('Different interpolation factor calculating ...')
                print('\n')
            else:
                if order in self.morder_finterp_lst:
                    print("this order has been calculated")
                    print('\n')
            moment_im = reconstruction.moment_im_with_finterp(self.filepath,
                                                              self.filename,
                                                              order, interp_num,
                                                              mvlength)
            # print(np.shape(moment_im))
            self.morder_finterp_lst.append(order)
            self.moments_finterp_set[order] = moment_im
            self.finterp_factor = interp_num
            return self.moments_finterp_set[order]

    def calc_moments_set(self, highest_order=4, frames=[],
                         mean_im=None, bleach_correction=False,
                         smooth_kernel=251, fbc=0.04):
        """
        Calculate moment-reconstructed images to the highest order.

        Parameters
        ----------
        highest_order : int
            The highest order number of moment-reconstructed images.
        frames : list of int
            The start and end frame number.
        mean_im : ndarray
            Average image of the tiff stack.
        bleach_correction : bool
            Whether to use bleaching correction.
        smooth_kernel : int, optional
            The size of the median filter window. Only used when bleach
            correction is True.
        fbc : float, optional
            The fraction of signal decrease within each block compared
            to the total signal decrease. Only used when bleach correction
            is True.

        Returns
        -------
        moments_set, moments_set_bc : dict
            order number (int) -> image (ndarray)
            A dictionary of calcualted moment-reconstructed images.
        """
        if bleach_correction is False:
            self.moments_set = reconstruction.calc_moments(self.filepath,
                                                           self.filename,
                                                           highest_order,
                                                           frames,
                                                           m_set={})
            self.morder = highest_order
            return self.moments_set
        else:
            self.moments_set_bc = reconstruction.block_ave_moments(
                self.filepath, self.filename,
                highest_order, smooth_kernel, fbc)
            self.fbc = fbc
            return self.moments_set_bc

    def cumulants_images(self, highest_order=4, frames=[],
                         m_set=None, bleach_correction=False,
                         smooth_kernel=251, fbc=0.04):
        """
        Calculate cumulant-reconstructed images to the highest order.

        Parameters
        ----------
        highest_order : int
            The highest order number of cumulant-reconstructed images.
        frames : list of int
            The start and end frame number.
        m_set : dict
            order number (int) -> image (ndarray)
            A dictionary of calcualted moment-reconstructed images.
        bleach_correction : bool
            Whether to use bleaching correction.
        smooth_kernel : int, optional
            The size of the median filter window. Only used when bleach
            correction is True.
        fbc : float, optional
            The fraction of signal decrease within each block compared
            to the total signal decrease. Only used when bleach correction
            is True.

        Returns
        -------
        cumulants_set, cumulants_set_bc : dict
            order number (int) -> image (ndarray)
            A dictionary of calcualted cumulant-reconstructed images.
        """
        if bleach_correction is False:
            self.corder = highest_order
            m_set = self.calc_moments_set(highest_order, frames)
            self.cumulants_set = reconstruction.calc_cumulants_from_moments(
                m_set)
            return self.cumulants_set
        else:
            self.cumulants_set_bc = reconstruction.block_ave_cumulants(
                self.filepath, self.filename,
                highest_order, smooth_kernel, fbc)
            return self.cumulants_set_bc

    def ldrc(self, order=6, window_size=[25, 25], mask_im=None, input_im=None):
        """
        Compress the dynamic range of moments-/cumulants reconstruced image
        with ldrc method. For details of ldrc, refer to ldrc.py and [1].

        Parameters
        ----------
        order : int
            The order of the reconstructed image.
        window_size : [int, int]
            The [x, y] dimension of the scanning window.
        mask_im : ndarray
            A reference image.
            Usually a average/sum image or second-order SOFI image is used.
        input_im : ndarray
            An input image, usually a high-order moment- or cumulant-
            reconstructed image.

        Returns
        -------
        ldrc_image : ndarray
            The compressed image with the same dimensions of input_im.
        """
        if input_im is None:
            if self.moments_set is not None:
                if self.morder >= order:
                    input_im = self.moments_set[order]
                else:
                    self.moment_image(order)
            else:
                self.moment_image(order)

        if mask_im is None:
            mask_im = self.average_image()

        self.ldrc_image = ldrc.ldrc(mask_im, input_im, order, window_size)
        return self.ldrc_image

    def deconvsk(self, est_psf=masks.gauss2d_mask((51, 51), 2), input_im=None,
                 deconv_lambda=1.5, deconv_iter=20):
        """
        Perform serial Richardson-Lucy deconvolution with a series of PSFs
        on the input image. For details of shrinking kernel deconvolution,
        refer to deconvsk.py and [1].

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
        deconv : ndarray
            Deconvoluted image.
        """
        if input_im is None:
            if self.moments_set is not None:
                input_im = self.moments_set[self.morder]
            elif self.ave is not None:
                input_im = self.ave
            else:
                input_im = self.average_image()
        self.deconv = deconvsk.deconvsk(est_psf, input_im,
                                        deconv_lambda, deconv_iter)
        return self.deconv

    def finterp_tiffstack(self, interp_num_lst=[2, 4], frames=[],
                          save_option=True, return_option=False):
        """
        Performs fourier interpolation on a tiff image (stack) with a list of
        interpolation factors (the number of pixels to be interpolated between
        two adjacent pixels).

        Parameters
        ----------
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
        finterps : list (ndarray)
            A list of interpolated image sereis corresponding to the interpolation
            factor list.
        """
        filename = self.filename[:-4]
        if return_option is False:
            f.fourier_interp_tiff(self.filepath, filename,
                                  interp_num_lst, frames,
                                  save_option, return_option)
        else:
            self.finterps = f.fourier_interp_tiff(self.filepath,
                                                  filename, interp_num_lst,
                                                  frames, save_option,
                                                  return_option)
            return self.finterps

    def finterp_image(self, input_im=None, interp_num_lst=[2, 4]):
        """Performs fourier interpolation on an image array."""
        if input_im is None:
            input_im = self.average_image()

        self.finterp = f.fourier_interp_array(input_im, interp_num_lst)
        return self.finterp

    def bleach_correct(self, fbc=0.04, smooth_kernel=251,
                       save_option=True, return_option=False):
        """
        Performs bleaching correction on a tiff image (stack) with a
        bleaching correction factor.

        Parameters
        ----------
        fbc : float
            The fraction of signal decrease within each block compared
            to the total signal decrease. Only used when bleach correction
            is True.
        smooth_kernel : int
            The size of the median filter window.
        save_option : bool
            Whether to save the corrected images into tiff files.
        return_option : bool
            Whether to return the corrected image series as a 3d array.

        Returns
        -------
        bc_im : ndarray
            All bleaching-corrected imagee in a 3d array.
        """
        if save_option is True:
            reconstruction.correct_bleaching(self.filepath, self.filename,
                                             fbc, smooth_kernel,
                                             save_option, return_option)
        if return_option is True:
            self.bc_ims = reconstruction.correct_bleaching(self.filepath,
                                                           self.filename,
                                                           fbc,
                                                           smooth_kernel,
                                                           save_option,
                                                           return_option)
            return self.bc_ims

    def add_filtered(self, image, filter_name='noise filter'):
        """Add (noise) filtered image as an attribute of the object."""
        self.filter_name = filter_name
        self.filtered = image

    def get_dims(self):
        """Get dimensions and frame number of the tiff video."""
        imstack = tiff.TiffFile(self.filepath + '/' + self.filename)
        n_frames = len(imstack.pages)
        xdim, ydim = np.shape(imstack.pages[0])
        return n_frames, xdim, ydim

    def add(self, **kwargs):
        """Add or update elements (attributes and/or dict entries). """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_frame(self, frame_num=0):
        """
        Get one frame of the tiff video.

        Parameters
        ----------
        frame_num : int
            The frame number (e.g. 0 for the first frame).

        Returns
        -------
        frame_im : ndarray
            Frame array.
        """
        if frame_num >= self.n_frames:
            raise Exception("'frame_num' exceeds the length of the video")
        if frame_num < 0 or np.int(frame_num) != frame_num:
            raise Exception("'frame_num' should be a non-negative integer")
        frame_im = tiff.imread(self.filepath + '/' +
                               self.filename, key=frame_num)
        return frame_im