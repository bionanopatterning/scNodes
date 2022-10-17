import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tifffile


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    # from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters?noredirect=1&lq=1
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)


def save_tiff(array, path, pixel_size_nm = 100, axes = "ZXY"):
    if not (path[-5:] == ".tiff" or path[-4:] == ".tif"):
        path += ".tif"
    if "/" in path:
        root = path.rfind("/")
        root = path[:root]
        if not os.path.exists(root):
            os.makedirs(root)
    tifffile.imwrite(path, array, resolution=(1./(1e-7 * pixel_size_nm), 1./(1e-7 * pixel_size_nm), 'CENTIMETER'))

def plot_histogram(data, bins = 'auto', title = None):
    plt.hist(data, bins = bins)
    if title:
        plt.title(title)
    plt.show()


def get_filetype(path):
    return path[path.rfind("."):]


def apply_lut_to_float_image(image, lut, contrast_lims = None):
    if len(image.shape) != 2:
        print("Image input in apply_lut_to_float_image is not 2D.")
        return False
    if isinstance(lut, list):
        _lut = np.asarray(lut)
    L, n = np.shape(_lut)
    if contrast_lims is None:
        contrast_lims = (np.amin(image), np.amax(image))
    image = (L-1) * (image - contrast_lims[0]) / (contrast_lims[1] - contrast_lims[0])
    image = image.astype(int)
    w, h = image.shape
    out_img = np.zeros((w, h, n))
    for x in range(w):
        out_img[x, :, :] = _lut[image[x, :], :]
    return out_img