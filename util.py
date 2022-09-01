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
    metadata = {'axes': axes}
    tifffile.imwrite(path, array, metadata=metadata, resolution = (1/(1e-3 * pixel_size_nm), 1/(1e-3 *pixel_size_nm), 'MICROMETER'))

def plot_histogram(data, bins = 'auto', title = None):
    plt.hist(data, bins = bins)
    if title:
        plt.title(title)
    plt.show()

def get_filetype(path):
    return path[path.rfind("."):]