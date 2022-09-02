from itertools import count
import numpy as np
import re
import glob
from PIL import Image
from skimage import transform

class Dataset:
    def __init__(self, path=None):
        """Path must be a path to an image in a folder containing 16-bit tif files - OR a tiffstack. All the imgs in the folder are part of the dataset"""
        self.path = path
        self.frames = list()
        self.n_frames = 0
        self.current_frame = 0
        self.pixel_size = 1
        self.directory = self.path[:self.path.rfind("/") + 1]
        self.load_data()
        self.img_width, self.img_height = self.get_indexed_image(0).load().shape

    def load_data(self):
        self.frames = list()

        def numerical_sort(value):
            numbers = re.compile(r'(\d+)')
            parts = numbers.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts

        # stack
        img = Image.open(self.path)
        if img.n_frames > 1:
            for i in range(img.n_frames):
                self.n_frames += 1
                self.frames.append(Frame(self.path, i))

        # folder
        else:
            files = sorted(glob.glob(self.directory + "*.tif*"), key=numerical_sort)
            for file in files:
                self.n_frames += 1
                self.frames.append(Frame(file))

    def get_indexed_image(self, index):
        if 0 <= index < self.n_frames:
            return self.frames[index]
        else:
            if index <= 0:
                return self.frames[0]
            else:
                return self.frames[-1]

    def get_active_image(self):
        return self.frames[self.current_frame]

    def __str__(self):
        return "Dataset object with source at: "+self.path+f"\nn_frames = {self.n_frames}"

class Frame:
    id_gen = count(1)

    def __init__(self, path, index=None):
        self.id = next(Frame.id_gen)
        self.index = index
        self.path = path
        self.data = None
        self.width = None
        self.height = None
        self.discard = False
        self.translation = [0.0, 0.0]
        self.maxima = list()

    def load(self):
        if self.data is not None:
            return self.data.copy()
        img = Image.open(self.path)
        if self.index is not None:
            img.seek(self.index)
        self.data = np.asarray(img).astype(np.float)
        self.width, self.height = self.data.shape
        return self.data.copy()

    def clean(self):
        self.translation = [0.0, 0.0]
        self.discard = False
        self.data = None
        self.maxima = list()

    def bake_transform(self):
        # TODO: generalize to use affine transformation matrix instead of just a translation vector.
        tmat = transform.AffineTransform(np.matrix([[1.0, 0.0, self.translation[0]], [0.0, 1.0, self.translation[1]], [0.0, 0.0, 1.0]]))
        self.data = transform.warp(self.data, tmat)

    def __str__(self):
        print(self.maxima)
        selfstr = "Frame at path: "+self.path + "\n" \
        + ("Discarded frame" if self.discard else "Frame in use") \
        + "Shift of ({self.translation[0]:.2f}, {self.translation[1]:.2f}) pixels detected."
        if self.maxima is not []:
            selfstr += f"\n{len(self.maxima)} particles found."
        return selfstr

