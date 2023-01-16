try:
    from . import finterp as f
    from . import filtering
    from . import switches as s
except:
    import finterp as f
    import filtering
    import switches as s

import numpy as np

if s.SPHINX_SWITCH is False:
    import tifffile as tiff

import scipy.special
import sys
import os
import collections
import math
import itertools


def sorted_k_partitions(seq, k):
    """
    Returns a list of all unique k-partitions of `seq`.
    Each partition is a list of parts, and each part is a tuple.
    """
    n = len(seq)
    groups = []

    def generate_partitions(i):
        if i >= n:
            yield list(map(tuple, groups))
        else:
            if n - i > k - len(groups):
                for group in groups:
                    group.append(seq[i])
                    yield from generate_partitions(i + 1)
                    group.pop()
            if len(groups) < k:
                groups.append([seq[i]])
                yield from generate_partitions(i + 1)
                groups.pop()

    result = generate_partitions(0)

    # Sort the parts in each partition in shortlex order
    result = [sorted(ps, key=lambda p: (len(p), p)) for ps in result]
    # Sort partitions by the length of each part, then lexicographically.
    result = sorted(result, key=lambda ps: (*map(len, ps), ps))
    # delete partitions with only 1 element
    result = [p for p in result if len(p[0]) > 1]

    return result


def average_image(filepath, filename, frames=[]):
    """
    Get the average image for a video file (tiff stack), either for the
    whole video or for user defined frames.

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    frames : list of int, optional
        Start and end frame number if not the whole video is used.

    Returns
    -------
    mean_im : ndarray
        The average image.
    """
    imstack = tiff.TiffFile(filepath + '/' + filename)
    xdim, ydim = np.shape(imstack.pages[0])
    mean_im = np.zeros((xdim, ydim))

    if not frames:
        frames = [0, len(imstack.pages)]
    for frame_num in range(frames[0], frames[1]):
        im = tiff.imread(filepath + '/' + filename, key=frame_num)
        mean_im = mean_im + im
        sys.stdout.write('\r')
        sys.stdout.write("frame " + str(frame_num) + '/' + str(frames[1]))
        sys.stdout.flush()
    mean_im = mean_im / (frames[1] - frames[0])
    imstack.close()

    return mean_im


def average_image_with_finterp(filepath, filename, interp_num):
    """
    Get the average image with fourier interpolation for a video file.

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    interp_num : int
        Interpolation factor.

    Returns
    -------
    mean_im : ndarray
        The average image after fourier interpolation. Interpolated
    images can be further used for SOFI processing.
    """
    original_mean_im = average_image(filepath, filename)
    finterp_mean_im = f.fourier_interp_array(original_mean_im, [interp_num])

    return finterp_mean_im[0]


def calc_moment_im(filepath, filename, order, frames=[], mean_im=None):
    """
    Get one moment-reconstructed image of a defined order for a video file.

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    order : int
        The order number of moment-reconstructed image.
    frames : list of int
        The start and end frame number.
    mean_im : ndarray
        Average image of the tiff stack.

    Returns
    -------
    moment_im : ndarray
        The moments-reconstructed image.
    """
    if mean_im is None:
        mean_im = average_image(filepath, filename, frames)
    imstack = tiff.TiffFile(filepath + '/' + filename)
    xdim, ydim = np.shape(imstack.pages[0])
    moment_im = np.zeros((xdim, ydim))
    print('Calculating the %s-order moment ...' %
          order)
    if frames:
        for frame_num in range(frames[0], frames[1]):
            im = tiff.imread(filepath + '/' + filename, key=frame_num)
            moment_im = moment_im + (im - mean_im) ** order
            sys.stdout.write('\r')
            sys.stdout.write("[{:{}}] {:.1f}%".format(
                "=" * int(30 / (frames[1] - frames[0]) * (frame_num - frames[0] + 1)), 29,
                (100 / (frames[1] - frames[0]) * (frame_num - frames[0] + 1))))
            sys.stdout.flush()
        moment_im = moment_im / (frames[1] - frames[0])
    else:
        mvlength = len(imstack.pages)
        for frame_num in range(mvlength):
            im = tiff.imread(filepath + '/' + filename, key=frame_num)
            moment_im = moment_im + (im - mean_im) ** order
            sys.stdout.write('\r')
            sys.stdout.write("[{:{}}] {:.1f}%".format(
                "=" * int(30 / mvlength * (frame_num + 1)), 29,
                (100 / mvlength * (frame_num + 1))))
            sys.stdout.flush()
        moment_im = moment_im / mvlength
    imstack.close()
    print('\n')
    return moment_im


def moment_im_with_finterp(filepath, filename, order, interp_num,
                           frames=[], mean_im=None):
    """
    Get one moment-reconstructed image of a defined order for a video file.

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    order : int
        The order number of moment-reconstructed image.
    interp_num : int
        The interpolation factor.
    mvlength : int
        The length of video for the reconstruction.
    mean_im : ndarray
        Average image of the tiff stack.

    Returns
    -------
    moment_im : ndarray
        The moments-reconstructed image.
    """
    if mean_im is None:
        mean_im = average_image_with_finterp(filepath, filename, interp_num)

    imstack = tiff.TiffFile(filepath + '/' + filename)
    xdim, ydim = np.shape(imstack.pages[0])
    moment_im = np.zeros(((xdim - 1) * interp_num + 1, (ydim - 1) * interp_num + 1))
    if frames:
        for frame_num in range(frames[0], frames[1]):
            im = tiff.imread(filepath + '/' + filename, key=frame_num)
            interp_im = f.fourier_interp_array(im, [interp_num])[0]
            moment_im = moment_im + (interp_im - mean_im) ** order
            sys.stdout.write('\r')
            sys.stdout.write("[{:{}}] {:.1f}%".format(
                "=" * int(30 / (frames[1] - frames[0]) * (frame_num - frames[0] + 1)), 29,
                (100 / (frames[1] - frames[0]) * (frame_num - frames[0] + 1))))
            sys.stdout.flush()
        moment_im = np.int64(moment_im / (frames[1] - frames[0]))
    else:
        mvlength = len(imstack.pages)
        for frame_num in range(mvlength):
            im = tiff.imread(filepath + '/' + filename, key=frame_num)
            interp_im = f.fourier_interp_array(im, [interp_num])[0]
            moment_im = moment_im + (interp_im - mean_im) ** order
            sys.stdout.write('\r')
            sys.stdout.write("[{:{}}] {:.1f}%".format(
                "=" * int(30 / mvlength * (frame_num + 1)), 29,
                (100 / mvlength * (frame_num + 1))))
            sys.stdout.flush()
        moment_im = np.int64(moment_im / mvlength)
    imstack.close()
    return moment_im


def calc_moments(filepath, filename, highest_order,
                 frames=None, m_set=None, mean_im=None):
    """
    Get all moment-reconstructed images to the user-defined highest order for
    a video file(tiff stack).

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    highest_order : int
        The highest order number of moment-reconstructed images.
    frames : list of int
        The start and end frame number.
    m_set : dict
        order number (int) -> image (ndarray)
        A dictionary of previously calcualted moment-reconstructed images.
    mean_im : ndarray
        Average image of the tiff stack.

    Returns
    -------
    m_set : dict
        order number (int) -> image (ndarray)
        A dictionary of calcualted moment-reconstructed images.
    """
    if m_set is None:
        m_set = {}
    if frames is None:
        frames = []
    if m_set:
        current_order = max(m_set.keys())
    else:
        current_order = 0

    # calculate average image (mean)
    if mean_im is None:
        mean_im = average_image(filepath, filename, frames)

    # load in the full image stack
    imstack = tiff.TiffFile(filepath + '/' + filename)

    # get the dimension of each frame
    xdim, ydim = np.shape(imstack.pages[0])

    def ordinal(n):
        return "%d%s" % (
            n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])

    order_lst = [ordinal(n + 1) for n in range(highest_order)]

    if highest_order > current_order:
        for order in range(current_order, highest_order):
            print('Calculating the %s-order moment...' %
                  order_lst[order])
            m_set[order + 1] = np.zeros((xdim, ydim))
            if frames:  # if the start and end frame number is specified.
                for frame_num in range(frames[0], frames[1]):
                    im = tiff.imread(filepath + '/' + filename, key=frame_num)
                    m_set[order + 1] = m_set[order + 1] + \
                                       np.power(im - mean_im, order + 1)
                    sys.stdout.write('\r')
                    sys.stdout.write("[{:{}}] {:.1f}%".format(
                        "=" * int(30 / (frames[1] - frames[0]) * (frame_num - frames[0] + 1)),
                        29, (100 / (frames[1] - frames[0]) * (frame_num - frames[0] + 1))))
                    sys.stdout.flush()
                m_set[order + 1] = np.int64(
                    m_set[order + 1] / (frames[1] - frames[0]))  # dvide by the total number of frames
                print('\n')
            else:
                mvlength = len(imstack.pages)
                for frame_num in range(mvlength):
                    im = tiff.imread(filepath + '/' + filename, key=frame_num)
                    m_set[order + 1] = m_set[order + 1] + \
                                       np.power(im - mean_im, order + 1)
                    sys.stdout.write('\r')
                    sys.stdout.write("[{:{}}] {:.1f}%".format(
                        "=" * int(30 / mvlength * (frame_num + 1)), 29,
                        (100 / mvlength * (frame_num + 1))))
                    sys.stdout.flush()
                m_set[order + 1] = np.int64(m_set[order + 1] / mvlength)
                print('\n')
    imstack.close()
    return m_set


def calc_cumulants_from_moments(moment_set):
    """
    Calculate cumulant images from moment images using the recursive relation.

    Parameters
    ----------
    moment_set : dict
        order number (int) -> image (ndarray)
        A dictionary of calcualted moment- images.

    Returns
    -------
    k_set : dict
        order number (int) -> image (ndarray)
        A dictionary of calcualted cumulant-reconstructed images.
    """
    if moment_set == {}:
        raise Exception("'moment_set' is empty.")

    k_set = {}
    highest_order = max(moment_set.keys())
    for order in range(1, highest_order + 1):
        k_set[order] = moment_set[order] - \
                       np.sum(np.array([scipy.special.comb(order - 1, i) * k_set[order - i] *
                                        moment_set[i] for i in range(1, order)]), axis=0)

    return k_set


def calc_block_moments(filepath, filename, highest_order, frames=[]):
    """
    Get moment-reconstructed images for user-defined frames (block) of
    a video file(tiff stack).

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    highest_order : int
        The highest order number of moment-reconstructed images.
    frames : list of int
        Start and end frame number.

    Returns
    -------
    m_set : dict
        order number (int) -> image (ndarray)
        A dictionary of calcualted moment-reconstructed images.

    Notes
    -----
    Similar to 'calc_moments'. Here we omit previously calculated m_set
    and mean_im as inputs since a block usually has much fewer number of
    frames and takes shorter calculation time.
    """
    mean_im = average_image(filepath, filename, frames)
    imstack = tiff.TiffFile(filepath + '/' + filename)
    if not frames:
        mvlength = len(imstack.pages)
        frames = [0, mvlength]
    xdim, ydim = np.shape(imstack.pages[0])
    block_length = frames[1] - frames[0]
    m_set = {}

    for order in range(highest_order):
        m_set[order + 1] = np.zeros((xdim, ydim))
        for frame_num in range(frames[0], frames[1]):
            im = tiff.imread(filepath + '/' + filename, key=frame_num)
            m_set[order + 1] = m_set[order + 1] + \
                               np.power(im - mean_im, order + 1)
        m_set[order + 1] = m_set[order + 1] / block_length
        sys.stdout.write('\r')
        sys.stdout.write("[{:{}}] {:.1f}%".format(
            "=" * int(30 / (highest_order - 1) * order), 29,
            (100 / (highest_order - 1) * order)))
        sys.stdout.flush()
    imstack.close()
    print('\n')
    return m_set


def calc_total_signal(filepath, filename):
    """
    Calculate the total signal intensity of each frame for the whole
    video (tiff stack).

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.

    Returns
    -------
    total_signal : 1darray
        Signal intensity of each frame ordered by the frame number.
    """
    imstack = tiff.TiffFile(filepath + '/' + filename)
    mvlength = len(imstack.pages)
    total_signal = np.zeros(mvlength)

    for frame_num in range(mvlength):
        im = tiff.imread(filepath + '/' + filename, key=frame_num)
        total_signal[frame_num] = sum(sum(im))
    imstack.close()
    return total_signal


def cut_frames(signal_level, fbc=0.04):
    """
    Find the list of frame number to cut the whole signal plot into seperate
    blocks based on the change of total signal intensity.

    Parameters
    ----------
    signal_level : 1darray
        Signal change over time (can be derived from 'calc_total_signal').
    fbc : float
        The fraction of signal decrease within each block compared to the
        total signal decrease.

    Returns
    -------
    bounds : list of int
        Signal intensities on the boundary of each block.
    frame_lst : list of int
        Frame number where to cut the whole signal plot into blocks.

    Notes
    -----
    The number of blocks is the inverse of the bleaching correction factor,
    fbc. For instance, if fbc=0.04, it means that in each block, the
    decrease in signal intensity is 4% if the total decrease, and the whole
    plot / video is cut into 25 blocks. For some data, it is possible that
    the maximun signal intensity does not appear at the beginning of the
    signal plot. Here, we consider all frames before the maximum in the
    same block as the maximum frame / intensity since usually the number
    of frames is not too large. The user can add in extra blocks if the
    initial intensity is much smaller than the maximum.
    """
    max_intensity, min_intensity = np.max(signal_level), np.min(signal_level)
    frame_num = np.argmax(signal_level)
    total_diff = max_intensity - min_intensity
    block_num = math.ceil(1 / fbc)
    frame_lst = []
    # lower bound of intensity for each block
    bounds = [int(max_intensity - total_diff * i * fbc)
              for i in range(1, block_num + 1)]
    i = 0
    while frame_num < len(signal_level) and i <= block_num:
        if signal_level[frame_num] < bounds[i]:
            frame_lst.append(frame_num)
            frame_num += 1
            i += 1
        else:
            frame_num += 1
    frame_lst = [0] + frame_lst + [len(signal_level)]
    bounds = [int(max_intensity)] + bounds

    return bounds, frame_lst


def min_image(filepath, filename, frames=[]):
    """
    Get the minimum image for a video file (tiff stack), either for the
    whole video or for user defined frames.

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    frames : list of int, optional
        Start and end frame number if not the whole video is used.

    Returns
    -------
    min_im : ndarray
        The minimum image.
    """
    imstack = tiff.TiffFile(filepath + '/' + filename)
    xdim, ydim = np.shape(imstack.pages[0])
    min_im = tiff.imread(filepath + '/' + filename, key=0)

    if not frames:
        frames = [0, len(imstack.pages)]
    for frame_num in range(frames[0], frames[1]):
        im = tiff.imread(filepath + '/' + filename, key=frame_num)
        min_im = (min_im <= im) * min_im + (min_im > im) * im
    imstack.close()
    return min_im


def correct_bleaching(filepath, filename, fbc=0.04, smooth_kernel=251,
                      save_option=True, return_option=False):
    """
    Performs bleaching correction on a tiff image (stack).

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
    # calculate total signal per frame as a function of time
    sig_b = calc_total_signal(filepath, filename)
    # filter the total signal to get a smoother bleaching profile
    filtered_sig_b = filtering.med_smooth(sig_b, kernel_size=251)
    # calculate the block boundary frame indexes based on the bleaching correction factor fbc
    bounds, frame_lst = cut_frames(filtered_sig_b, fbc=fbc)
    # obtain the total number of blocks.
    block_num = math.ceil(1 / fbc)

    # loop over all the blocks
    for i in range(block_num):
        # calculate the average image for each block
        ave_im = average_image(filepath, filename,
                               frames=[frame_lst[i], frame_lst[i + 1]])
        # loop over every image within the block
        for frame_num in range(frame_lst[i], frame_lst[i + 1]):
            im = tiff.imread(filepath + '/' + filename, key=frame_num)  # read out the image
            norm_im = im - ave_im  # subtract the average from the image
            norm_im = np.int_(np.around(norm_im))  # covnert to integer
            tiff.imwrite(filepath + '/' + filename[:-4] + '_bc0.tif',
                         norm_im, dtype='int', append=True)  # write out a temporariy tiff stack.
            # note that there can be negative values in the image because this is after mean-subtraction

    # store the filename of this temporary tiff stack.
    filename = filename[:-4] + '_bc0.tif'
    # obtain the minimum value across the entire temporary tiff stack (which will be a negative number)
    min_im = min_image(filepath, filename)
    # prepare return value variable
    imstack = tiff.TiffFile(filepath + '/' + filename)
    xdim, ydim = np.shape(imstack.pages[0])
    mvlength = len(imstack.pages)
    if return_option is True:
        bc_im = np.zeros((mvlength, xdim, ydim))
    imstack.close()
    # now need to shift the image pixel values up to set the minimum value to be zero in the following for-loop
    for i in range(mvlength):
        im = tiff.imread(filepath + '/' + filename, key=i)
        norm_im = im - min_im
        norm_im = np.int_(np.around(norm_im))
        if save_option is True:
            tiff.imwrite(filepath + '/' + filename[:-8] + '_bc.tif',
                         norm_im, dtype='int', append=True)
        if return_option is True:
            bc_im[i] = norm_im
    # delete the temperary tiff.
    os.remove(filepath + '/' + filename)
    if return_option is True:
        return bc_im


def moments_all_blocks(filepath, filename,
                       highest_order, smooth_kernel=251, fbc=0.04):
    """
    Get moment-reconstructed images for seperate blocks with user-defined
    noise filtering and bleaching correction factor (fbc).

    Within each block, the amount of signal decrease is identical. This
    amount of signal decrease is characterized by the bleaching correction
    factor, fbc, which is the fractional signal decrease within each block
    (as compared to the total signal decrease over the whole signal plot).

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    highest_order : int
        The highest order number of moment-reconstructed images.
    smooth_kernel : int
        The size of the median filter ('filtering.med_smooth') window.
    fbc : float
        The fraction of signal decrease within each block compared to the
        total signal decrease.

    Returns
    -------
    m_set_all_blocks : dict
        block_number (int) -> {order number (int) -> image (ndarray)}
        A dictionary of moment-reconstructed images of each block.

    Notes
    -----
    Similar to 'calc_moments'. Here we omit previously calculated m_set
    and mean_im as inputs since a block usually has much fewer number of
    frames and takes shorter calculation time.
    """
    all_signal = calc_total_signal(filepath, filename)
    filtered_signal = filtering.med_smooth(all_signal, smooth_kernel)
    _, cut_frame = cut_frames(filtered_signal, fbc)
    block_num = math.ceil(1 / fbc)
    m_set_all_blocks = {}
    for i in range(block_num):
        print('Calculating moments of block %d...' % i)
        m_set_all_blocks[i] = calc_block_moments(filepath,
                                                 filename,
                                                 highest_order,
                                                 [cut_frame[i],
                                                  cut_frame[i + 1]])

    return m_set_all_blocks


def cumulants_all_blocks(m_set_all_blocks):
    """
    Calculate cumulant-reconstructed images from moment-reconstructed images
    of each block. Similar to 'calc_cumulants_from_moments'.
    """
    if m_set_all_blocks == {}:
        raise Exception("'moment_set' is empty.")

    k_set_all_blocks = {i: calc_cumulants_from_moments(m_set_all_blocks[i])
                        for i in m_set_all_blocks}

    return k_set_all_blocks


def block_ave_cumulants(filepath, filename,
                        highest_order, smooth_kernel=251, fbc=0.04):
    """
    Get average cumulant-reconstructed images of all blocks determined by
    user-defined noise filtering and bleaching correction factor (fbc).

    Within each block, the amount of signal decrease is identical. This
    amount of signal decrease is characterized by the bleaching correction
    factor, fbc, which is the fractional signal decrease within each block
    (as compared to the total signal decrease over the whole signal plot).

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    highest_order : int
        The highest order number of moment-reconstructed images.
    smooth_kernel : int
        The size of the median filter ('filtering.med_smooth') window.
    fbc : float
        The fraction of signal decrease within each block compared to the
        total signal decrease.

    Returns
    -------
    ave_k_set : dict
        order number (int) -> image (ndarray)
        A dictionary of avereage cumulant-reconstructed images of all blocks.

    Notes
    -----
    For more information on noise filtering and bleaching corrextion, please
    see appendix 3 of [1].

    References
    ----------
    .. [1] X. Yi, and S. Weiss. "Cusp-artifacts in high order superresolution
    optical fluctuation imaging." bioRxiv: 545574 (2019).
    """
    k_set_all_blocks = cumulants_all_blocks(
        moments_all_blocks(filepath, filename,
                           highest_order,
                           smooth_kernel, fbc))
    block_num = len(k_set_all_blocks)
    k_set_lst = list(k_set_all_blocks.values())
    counter = collections.Counter()
    for d in k_set_lst:
        counter.update(d)
    ave_k_set = dict(counter)
    ave_k_set = {i: ave_k_set[i] / block_num for i in ave_k_set}

    return ave_k_set


def block_ave_moments(filepath, filename,
                      highest_order, smooth_kernel=251, fbc=0.04):
    """
    Get average moment-reconstructed images of all blocks determined by
    user-defined noise filtering and bleaching correction factor (fbc).
    Similar to 'block_ave_cumulants'.

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    highest_order : int
        The highest order number of moment-reconstructed images.
    smooth_kernel : int
        The size of the median filter ('filtering.med_smooth') window.
    fbc : float
        The fraction of signal decrease within each block compared to the
        total signal decrease.

    Returns
    -------
    ave_m_set : dict
        order number (int) -> image (ndarray)
        A dictionary of avereage moment-reconstructed images of all blocks.
    """
    m_set_all_blocks = moments_all_blocks(filepath, filename,
                                          highest_order, smooth_kernel, fbc)
    block_num = len(m_set_all_blocks)
    m_set_lst = list(m_set_all_blocks.values())
    counter = collections.Counter()
    for d in m_set_lst:
        counter.update(d)
    ave_m_set = dict(counter)
    ave_m_set = {i: ave_m_set[i] / block_num for i in ave_m_set}

    return ave_m_set


def calc_xc_im(filepath, filename, ri_range, frames=[]):
    """
    Get cross-cumulant image with different pixel combinations.

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    ri_range : int
        Maximium distance (# pixels) between the original the selected pixel.
    frames : list of int
        The start and end frame number.

    Returns
    -------
    xc : dict
        pixel (tuple) -> image (ndarray)
        A dictionary of reconstructed images with pixel index as keys.
    """
    series_length = ((2 * ri_range + 1) ** 2 - 1) // 2
    ri_series = [(i, j) for i in range(-ri_range, 0)
                 for j in range(-ri_range, ri_range + 1)] + \
                [(0, i) for i in range(-ri_range, 0)]
    imstack = tiff.TiffFile(filepath + '/' + filename)
    xdim, ydim = np.shape(imstack.pages[0])
    im_mean = average_image(filepath, filename, frames)
    imI, imJ = {}, {}
    xc = {ri_series[i]: np.zeros((xdim - 2 * ri_range, ydim - 2 * ri_range))
          for i in range(series_length)}
    if frames:
        mvlength = frames[1] - frames[0]
        for frame_num in range(frames[0], frames[1]):
            im = tiff.imread(filepath + '/' + filename, key=frame_num) - \
                 im_mean
            for series, xc_im in xc.items():
                imI[series] = im[ri_range + series[0]:xdim - ri_range + series[0],
                              ri_range + series[1]:ydim - ri_range + series[1]]
                imJ[series] = im[ri_range - series[0]:xdim - ri_range - series[0],
                              ri_range - series[1]:ydim - ri_range - series[1]]
                xc[series] = xc_im + np.multiply(imI[series], imJ[series])
    else:
        mvlength = len(imstack.pages)
        for frame_num in range(mvlength):
            im = tiff.imread(filepath + '/' + filename, key=frame_num) - \
                 im_mean
            for series, xc_im in xc.items():
                imI[series] = im[ri_range + series[0]:xdim - ri_range + series[0],
                              ri_range + series[1]:ydim - ri_range + series[1]]
                imJ[series] = im[ri_range - series[0]:xdim - ri_range - series[0],
                              ri_range - series[1]:ydim - ri_range - series[1]]
                xc[series] = xc_im + np.multiply(imI[series], imJ[series])
    for series, xc_im in xc.items():
        xc[series] = xc_im / mvlength

    return xc


def calc_moments_with_lag(filepath, filename, tauSeries, frames=[]):
    """
    Get moment-reconnstructed images with defined time lags.

    Parameters
    ----------
    filepath : str
        Path to the tiff file.
    filename : str
        Name of the tiff file.
    tauSeries : list of int
        A list of time lags for frames contribute to moment reconstruction.
        The first element is recommended to be 0, and there should seven
        elements in the list.
    frames : list of int
        The start and end frame number.

    Returns
    -------
    m_set : dict
        order (int) -> dict (partition (tuple) -> image (ndarray))
        A nested dictionary of moment-reconstructions of all partitions.
    """
    if len(tauSeries) < 2:
        raise Exception("The number of time lags should be more than one.")
    if tauSeries[0] != 0:
        raise Exception("The first time lag should be zero.")
    max_order = len(tauSeries)
    mean_im = average_image(filepath, filename, frames)
    xdim, ydim = np.shape(mean_im)

    # define frame range based on tauSeries
    if frames:
        start_frame, end_frame = frames[0], frames[1] - max(tauSeries)
        mvlength = end_frame - start_frame
    else:
        imstack = tiff.TiffFile(filepath + '/' + filename)
        mvlength = len(imstack.pages)
        start_frame, end_frame = 0, mvlength - max(tauSeries)
        mvlength = end_frame - start_frame

    # calculate partitions and their combinations
    seq = list(range(len(tauSeries)))
    if max_order > 3:
        partitions = {element_num: [subset
                                    for subset in itertools.combinations(seq, element_num)]
                      for element_num in range(2, max_order - 1)}

    # set up dictionaries and initialize with zeros array
    m_set = {order: {} for order in range(2, max_order + 1)}
    for order in range(2, max_order + 1):
        m_set[order][tuple(range(0, order))] = np.zeros((xdim, ydim))
    if max_order > 3:
        for order in range(2, max_order - 1):
            m_set[order] = {partitions[order][i]: np.zeros(
                (xdim, ydim)) for i in range(len(partitions[order]))}

    # calculate moments
    for frame_num in range(start_frame, end_frame):
        if max_order > 3:
            # 1. partitions
            for order in range(2, max_order - 1):
                for im_group in partitions[order]:
                    imSeries = np.array([tiff.imread(
                        filepath + '/' + filename,
                        key=frame_num + tauSeries[i]) - mean_im
                                         for i in im_group])
                    m_set[order][im_group] = m_set[order][im_group] + \
                                             np.prod(imSeries, axis=0) / mvlength
            # 2. the whole lag series
            for order in range(max_order - 1, max_order + 1):
                imSeries = np.array([tiff.imread(
                    filepath + '/' + filename, key=frame_num + i) - mean_im
                                     for i in tauSeries])
                m_set[order][tuple(range(0, order))] = m_set[order][tuple(
                    range(0, order))] + \
                                                       np.prod(imSeries[:order], axis=0) / mvlength
        else:
            # only the whole lag series
            imSeries = np.array([tiff.imread(
                filepath + '/' + filename, key=frame_num + i) - mean_im
                                 for i in tauSeries])
            for order in range(2, max_order + 1):
                m_set[order][tuple(range(0, order))] = m_set[order][tuple(
                    range(0, order))] + \
                                                       np.prod(imSeries[:order], axis=0) / mvlength

    return m_set


def calc_cumulants_from_moments_with_lag(m_set, tauSeries):
    """
    Calculate cumulant-reconstructed images from moment-reconstructed images
    with time lags.

    Parameters
    ----------
    m_set : dict
        order (int) -> dict (partition (tuple) -> image (ndarray))
        A nested dictionary of moment-reconstructions of all partitions that
        calculated from 'calc_moments_with_lag'.
    tauSeries : list of int
        A list of time lags for frames contribute to moment reconstruction.
        The first element is recommended to be 0, and there should be seven
        elements in the list. The values should be the same as the input for
        'calc_moments_with_lag'.

    Returns
    -------
    k_set : dict
        order number (int) -> image (ndarray)
        A dictionary of calcualted cumulant-reconstructed images with time
        lags.
    """
    k_set = {}
    seq = list(range(len(tauSeries)))
    max_order = len(tauSeries)
    for order in range(2, max_order + 1):
        k_set[order] = m_set[order][tuple(range(0, order))]
    if max_order > 3:
        for order in range(4, max_order + 1):
            seq = list(range(order))
            max_partition_num = max_order // 2
            for partition_num in range(2, max_partition_num + 1):
                partition_comb = sorted_k_partitions(seq, partition_num)
                for part in partition_comb:
                    k_set[order] = k_set[order] + \
                                   (-1) ** (partition_num - 1) * \
                                   np.math.factorial(partition_num - 1) * \
                                   np.prod([m_set[len(i)][i] for i in part],
                                           axis=0)

    return k_set