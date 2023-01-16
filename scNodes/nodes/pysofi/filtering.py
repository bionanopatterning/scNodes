import sys

sys.path.append('')
import numpy as np

try:
    from . import masks
except:
    import masks

from scipy import signal


def filter1d_same(time_series, noise_filter):
    """
    Filter original time_series with noise_filter, and return the 
    filtered_series with the same length as the original sereis.
    Compared to MATLAB results, when the length of the filter and 
    frame number are both even, the filtered result would shift to 
    left by one number. In other cases, results are the same.
    """
    filtered_series = np.convolve(time_series, noise_filter, 'full')
    filtered_center = len(filtered_series) // 2
    original_center = len(time_series) / 2
    filtered_series = filtered_series[np.int(filtered_center -
                                             np.floor(original_center)):
                                      np.int(filtered_center +
                                             np.ceil(original_center))]
    return filtered_series


def noise_filter1d(dset, im_set, noise_filter=[],
                   filtername='noise filter after M6',
                   filenames=None, return_option=False, return_type='dict'):
    """
    Perform noise filtering on a image stack along the time axis for each 
    pixel independently.

    Parameters
    ----------
    dset : dict
        filename (str) -> Data (object).
        A dictionary mapping tiff stack filenames to Data object.
    im_set : dict
        filename (str) -> pre-filtering image (ndarray).
        A dictionary mapping tiff filenames to images need to be filtered.
    noise_filter : ndarray
        Noise filtering kernel, e.g. 1D-Gaussian.
    filtername : str
        Name of the filter for Data.add_filtered. 
    filenames : list (str)
        Sequence of filenames for the filtering.
    return_option : bool
        Whether to return the results after noise filtering or not.
    return_type : str
        Choose the format of the returned value. Can use either 'ndarray' or 'dict'

    Returns
    -------
    m_filtered : ndarray
        Filtered image stack in the format of a numpy.ndarray.
    m_filtered : dict
        Filtered image stack in the format of a dictionary.

    """
    if filenames is None:
        filenames = [*im_set]

    if len(noise_filter) == 0:
        noise_filter = masks.gauss1d_mask((1, 21), 2)

    def get_series(i, j):
        return [im_set[filename][i, j]
                for filename in filenames]

    xdim, ydim = np.shape(im_set[filenames[0]])
    m_length = len(filenames)
    m_filtered = np.zeros((m_length, xdim, ydim))
    print('Noise filtering ...')
    for i in range(xdim):
        for j in range(ydim):
            time_series = get_series(i, j)
            m_filtered[:, i, j] = filter1d_same(time_series, noise_filter)
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * (i + 1) / xdim),
                                           100 * (i + 1) / xdim))
        sys.stdout.flush()
    for k in range(m_length):
        dset[filenames[k]].add_filtered(m_filtered[k], filtername)

    m_filtered_set = {}
    for k in np.arange(len([*im_set])):
        m_filtered_set[[*im_set][k]] = m_filtered[k, :, :]

    if return_option is True:
        if return_type == 'ndarray':
            return m_filtered
        if return_type == 'dict':
            return m_filtered_set


def med_smooth(ori_signal, kernel_size=251):
    """
    Perform a one-dimensional median filter with 'reflect' padding.
    For more information, please check scipy.signal.medfilt.
    """
    signal_pad = np.append(np.append(ori_signal[0:kernel_size][::-1],
                                     ori_signal),
                           ori_signal[-kernel_size:][::-1])
    filtered_signal = signal.medfilt(signal_pad, kernel_size)
    return filtered_signal[kernel_size:-kernel_size]