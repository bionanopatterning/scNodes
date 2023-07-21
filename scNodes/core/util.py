import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import glob
from scNodes.core import config as cfg
import time
import mrcfile
from scipy.ndimage import label

timer = 0.0

def coords_from_tsv(coords_path):
    coords = []
    with open(coords_path, 'r') as file:
        for line in file:
            x, y, z = map(int, line.strip().split('\t'))
            coords.append((x, y, z))
    return coords

def extract_particles(vol_path, coords_path, boxsize, two_dimensional=False):
    coords = coords_from_tsv(coords_path)
    data = mrcfile.mmap(vol_path, mode='r').data
    imgs = list()

    d = boxsize // 2
    for p in coords:
        x, y, z = p
        if two_dimensional:
            imgs.append(data[z, y - d:y + d, x - d:x + d])
        else:
            imgs.append(data[z - d:z + d, y - d:y + d, x - d:x + d])
    return imgs

def get_maxima_3d(mrcpath, threshold=128, min_volume=None, min_weight=None, min_spacing=10.0, save_txt=True, sort_by_weight=True):
    """
    min_volume: minimum volume of selected blobs, in units of nm**3
    min_spacing: minimum spacing between selected blobs, in nm
    """
    data = mrcfile.read(mrcpath)
    Z, Y, X = data.shape

    print(f"\nLoading {mrcpath}")
    pixel_size = mrcfile.open(mrcpath, header_only=True).voxel_size.x / 10.0
    print(f"Pixel size is {pixel_size} nm.")

    # threshold data
    print(f"Thresholding at {threshold}")
    mask = data > threshold

    # label contiguous nonzero regions
    print("Applying scipy.ndimage.label & indexing blobs")
    labels, N = label(mask)
    Z, Y, X = np.nonzero(labels)

    # parse blobs
    blobs = dict()
    for i in range(len(X)):
        z = Z[i]
        y = Y[i]
        x = X[i]

        l = labels[z, y, x]
        if l not in blobs:
            blobs[l] = Blob()

        blobs[l].x.append(x)
        blobs[l].y.append(y)
        blobs[l].z.append(z)
        blobs[l].v.append(data[z, y, x])

    if min_volume:
        to_pop = list()
        for key in blobs:
            volume = blobs[key].get_volume()
            if (pixel_size ** 3 * volume) < min_volume:
                to_pop.append(key)

        for key in to_pop:
            blobs.pop(key)
        print(f"Removing {len(to_pop)} blobs because they are too small. N = {len(blobs)} blobs remaining.")

    if min_weight:
        to_pop = list()
        for key in blobs:
            weight = blobs[key].get_weight()
            if weight < min_weight:
                to_pop.append(key)
        for key in to_pop:
            blobs.pop(key)
        print(f"Removing {len(to_pop)} blobs because their sum weight is too little. N = {len(blobs)} blobs remaining.")

    blobs = list(blobs.values())
    metrics = list()
    print(f"Sorting blobs by {'summed prediction weight' if sort_by_weight else 'volume'}.")
    for blob in blobs:
        if sort_by_weight:
            metrics.append(blob.get_weight())
        else:
            metrics.append(blob.get_volume())

    indices = np.argsort(metrics)[::-1]
    coordinates = list()
    for i in indices:
        coordinates.append(blobs[i].get_centroid())

    # remove points that are too close to others.
    remove = list()
    i = 0
    while i < len(coordinates):
        for j in range(0, i):
            if i in remove:
                continue
            p = np.array(coordinates[i])
            q = np.array(coordinates[j])
            d = np.sum((p - q) ** 2) ** 0.5 * pixel_size
            if d < min_spacing:
                remove.append(j)
        i += 1

    print(f"Removing N = {len(remove)} blobs due to proximity to better blobs.")
    for i in reversed(remove):
        coordinates.pop(i)

    if not save_txt:
        print(f"Found N = {len(coordinates)} blobs.")
        return len(coordinates)

    out_path = mrcpath[:-4] + "_coords.txt"
    print(f"Converting the final N = {len(coordinates)} to integers and saving to file: {out_path}")
    with open(out_path, 'w') as out_file:
        for i in range(len(coordinates)):
            x = int(coordinates[i][0])
            y = int(coordinates[i][1])
            z = int(coordinates[i][2])
            out_file.write(f"{x}\t{y}\t{z}\n")

    return len(coordinates)


class Blob:
    def __init__(self):
        self.x = list()
        self.y = list()
        self.z = list()
        self.v = list()

    def get_centroid(self):
        return np.mean(self.x), np.mean(self.y), np.mean(self.z)

    def get_center_of_mass(self):
        mx = np.sum(np.array(self.x) * np.array(self.v))
        my = np.sum(np.array(self.y) * np.array(self.v))
        mz = np.sum(np.array(self.z) * np.array(self.v))
        m = self.get_weight()
        return mx / m, my / m, mz / m

    def get_volume(self):
        return len(self.x)

    def get_weight(self):
        return np.sum(self.v)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
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


def save_tiff(array, path, pixel_size_nm=100, axes="ZXY"):
    if not (path[-5:] == ".tiff" or path[-4:] == ".tif"):
        path += ".tif"
    if "/" in path:
        root = path.rfind("/")
        root = path[:root]
        if not os.path.exists(root):
            os.makedirs(root)
    tifffile.imwrite(path, array.astype(np.float32), resolution=(
        1. / (1e-7 * pixel_size_nm), 1. / (1e-7 * pixel_size_nm),
        'CENTIMETER'))  # Changed to astype npfloat32 on 230105 to fix importing tifffile tiff with PIL Image open. Default for tifffile export unspecified-float np array appears to be 64bit which PIL doesnt support.


def save_png(array, path, alpha=True):
    try:
        if not path[-4:] == '.png':
            path += '.png'
        if "/" in path:
            root = path.rfind("/")
            root = path[:root]
            if not os.path.exists(root):
                os.makedirs(root)
        if array.dtype != np.dtype(np.uint8):
            array = array.astype(np.uint8)
        if alpha:
            Image.fromarray(array, mode="RGBA").save(path)
        else:
            Image.fromarray(array, mode="RGB").save(path)
    except Exception as e:
        cfg.set_error(e, "Error exporting image as .png. Is the path valid?")


def plot_histogram(data, bins='auto', title=None):
    plt.hist(data, bins=bins)
    if title:
        plt.title(title)
    plt.show()


def get_filetype(path):
    return path[path.rfind("."):]


def apply_lut_to_float_image(image, lut, contrast_lims=None):
    if len(image.shape) != 2:
        print("Image input in apply_lut_to_float_image is not 2D.")
        return False
    if isinstance(lut, list):
        _lut = np.asarray(lut)
    L, n = np.shape(_lut)
    if contrast_lims is None:
        contrast_lims = (np.amin(image), np.amax(image))
    image = (L - 1) * (image - contrast_lims[0]) / (contrast_lims[1] - contrast_lims[0])
    image = image.astype(int)
    w, h = image.shape
    out_img = np.zeros((w, h, n))
    for x in range(w):
        out_img[x, :, :] = _lut[image[x, :], :]
    return out_img


def is_path_tiff(path):
    if path[-4:] == ".tif" or path[-5:] == ".tiff":
        return True
    return False


def load(path, tag=None):
    if is_path_tiff(path):
        return loadtiff(path)
    else:
        return loadfolder(path, tag)


def loadfolder(path, tag=None):
    if path[-1] != "/":
        path += "/"
    pre_paths = glob.glob(path + "*.tiff") + glob.glob(path + "*.tif")
    paths = list()
    if tag is not None:
        for path in pre_paths:
            if tag in path:
                paths.append(path)
    else:
        paths = pre_paths
    _data = Image.open(paths[0])
    _frame = np.asarray(_data)

    width = _frame.shape[0]
    height = _frame.shape[1]
    depth = len(paths)
    data = np.zeros((depth, width, height), dtype=np.float32)
    _n = len(paths)
    for f in range(0, _n):
        printProgressBar(f, _n, prefix="Loading tiff file\t", length=30, printEnd="\r")
        _data = Image.open(paths[f])
        data[f, :, :] = np.asarray(_data)
    return data


def loadtiff(path):
    "Loads stack as 3d array with dimensions (frames, width, height)"
    _data = Image.open(path)
    _frame = np.asarray(_data)

    width = _frame.shape[0]
    height = _frame.shape[1]
    depth = _data.n_frames
    data = np.zeros((depth, width, height), dtype=np.int16)
    for f in range(0, depth):
        printProgressBar(f, depth - 1, prefix="Loading tiff file\t", length=30, printEnd="\r")
        _data.seek(f)
        data[f, :, :] = np.asarray(_data)
    data = np.squeeze(data)
    return data


def tic():
    global timer
    timer = time.time_ns()


def toc(msg):
    print(msg + f" {(time.time_ns() - timer) * 1e-9:.3} seconds")


def clamp(a, _min, _max):
    return min(max(a, _min), _max)
