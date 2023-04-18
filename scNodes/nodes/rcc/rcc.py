import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

# Redundant cross correlation (RCC):
# Wang et al.  'Localization events-based sample drift correction for localization microscopy with redundant cross-correlation algorithm' (2014), Optics Express. DOI: 10.1364/OE.22.015982

# Code partially based on the Picasso python implementation of RCC (at https://github.com/jungmannlab/picasso)
# by Schnitzbauer et al., in 'Super-resolution microscopy with DNA-PAINT' (2017) Nature Protocols. DOI: 10.1038/nprot.2017.024

def xcorr(f, g):
    F = np.fft.fft2(f)
    G_star = np.conj(np.fft.fft2(g))
    return np.fft.fftshift(np.real(np.fft.ifft2(F * G_star)))


def find_peak_in_xcorr(xcorr, fit_roi=5):
    if xcorr.shape[0] < (fit_roi * 2 + 1) or xcorr.shape[1] < (fit_roi * 2 + 1):
        raise Exception("fit_roi too large: fit_roi * 2 + 1 > xcorr shape.")
    x, y = np.unravel_index(np.argmax(xcorr), xcorr.shape)

    x_range = [x - fit_roi, x + fit_roi + 1]
    y_range = [y - fit_roi, y + fit_roi + 1]
    if x_range[0] < 0:
        x_range = [0, 2 * fit_roi + 1]
    elif x_range[1] > xcorr.shape[0]:
        x_range = [xcorr.shape[0] - 2 * fit_roi - 1, xcorr.shape[0]]
    if y_range[0] < 0:
        y_range = [0, 2 * fit_roi + 1]
    elif y_range[1] > xcorr.shape[1]:
        y_range = [xcorr.shape[1] - 2 * fit_roi - 1, xcorr.shape[1]]
    # crop:
    crop = xcorr[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    # centroid:
    mass = 0
    moment_x = 0
    moment_y = 0
    for _x in range(fit_roi * 2 + 1):
        for _y in range(fit_roi * 2 + 1):
            m = crop[_x, _y]
            moment_x += _x * m
            moment_y += _y * m
            mass += m
    com_x = moment_x / mass
    com_y = moment_y / mass
    x = com_x + x_range[0] - xcorr.shape[0] // 2
    y = com_y + y_range[0] - xcorr.shape[1] // 2
    return [x, y]


def rcc(particle_data, segments=10, pixel_size=200.0):
    x = particle_data.parameters["x [nm]"]
    y = particle_data.parameters["y [nm]"]
    f = particle_data.parameters["frame"]

    x_min = np.amin(x)
    x_max = np.amax(x)
    y_min = np.amin(y)
    y_max = np.amax(y)
    x_range = [x_min, x_max - (x_max - x_min) % pixel_size]
    y_range = [y_min, y_max - (y_max - y_min) % pixel_size]
    n_bins_x = int((x_range[1] - x_range[0]) / pixel_size)
    n_bins_y = int((y_range[1] - y_range[0]) / pixel_size)
    n_frames = int(np.amax(f))

    starting_indices = (np.linspace(0, 1, segments+1) * n_frames).astype(int)
    images = list()
    for i in range(segments):
        start = starting_indices[i]
        stop = starting_indices[i+1]

        f_mask = (start <= f) * (f < stop)
        _x = x[f_mask]
        _y = y[f_mask]

        # generate an image
        img, _, _, _ = plt.hist2d(_x, _y, bins=[n_bins_x, n_bins_y], range=[x_range, y_range])
        plt.close()
        images.append(img)

    # correlate the images
    shifts = np.zeros((segments, segments, 2))
    for i in range(segments):
        for j in range(i+1, segments):
            # compute the image shift for these two segments
            corr = xcorr(images[i], images[j])
            shifts[i, j] = find_peak_in_xcorr(corr)

    # compute the drift for every segment - code below in particular based on Picasso (see top of this file for reference)
    N = int(segments * (segments - 1) / 2)
    rij = np.zeros((N, 2))
    A = np.zeros((N, segments - 1))
    flag = 0
    for i in range(segments - 1):
        for j in range(i + 1, segments):
            rij[flag, 0] = shifts[i, j, 1]
            rij[flag, 1] = shifts[i, j, 0]
            A[flag, i:j] = 1
            flag += 1
    Dj = np.dot(np.linalg.pinv(A), rij)
    shift_y = np.insert(np.cumsum(Dj[:, 0]), 0, 0)
    shift_x = np.insert(np.cumsum(Dj[:, 1]), 0, 0)

    # interpolate these timepoints to find drift for every frame.
    t = (starting_indices[1:] + starting_indices[:-1]) / 2
    all_shift_x = InterpolatedUnivariateSpline(t, shift_x, k=3)
    all_shift_y = InterpolatedUnivariateSpline(t, shift_y, k=3)
    t_inter = np.arange(n_frames + 1)

    drift = (all_shift_x(t_inter) * pixel_size, all_shift_y(t_inter) * pixel_size)
    return drift
