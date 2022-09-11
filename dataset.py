from itertools import count
import numpy as np
import re
import glob
from PIL import Image
from skimage import transform
import pandas as pd
import copy


class Dataset:
    idgen = count(1)

    def __init__(self, path=None, pixel_size=100):
        self.id = next(Dataset.idgen)
        """Path must be a path to an image in a folder containing 16-bit tif files - OR a tiffstack. All the imgs in the folder are part of the dataset"""
        self.path = path
        self.frames = [[]]
        self.n_frames = 0
        self.current_frame = 0
        self.pixel_size = pixel_size
        self.initialized = False
        self.reconstruction_roi = [0, 0, 0, 0]
        if self.path is not None:
            self.load_data()
            self.directory = self.path[:self.path.rfind("/") + 1]
            self.img_width, self.img_height = self.get_indexed_image(0).load().shape
            self.initialized = True
        else:
            self.directory = ""
            self.img_width, self.img_height = (0, 0)

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
            i = 0
            files = sorted(glob.glob(self.directory + "*.tif*"), key=numerical_sort)
            for file in files:
                self.n_frames += 1
                self.frames.append(Frame(file, i))
                i += 1

        self.set_pixel_size(self.pixel_size)

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

    def set_pixel_size(self, pixel_size):
        self.pixel_size = pixel_size
        for frame in self.frames:
            frame.pixel_size = self.pixel_size

    def __str__(self):
        return "Dataset object with source at: "+self.path+f"\nn_frames = {self.n_frames}"


class Frame:
    id_gen = count(1)

    def __init__(self, path, index=None):
        self.id = next(Frame.id_gen)
        self.index = index
        self.path = path
        self.data = None
        self.raw_data = None
        self.width = None
        self.height = None
        self.discard = False
        self.translation = [0.0, 0.0]
        self.maxima = list()
        self.particles = list()
        self.pixel_size = 1

    def load(self):
        if self.data is not None:
            print(f"Returning preloaded copy of frame {self.id}")
            return self.data
        elif self.raw_data is not None:
            print(f"Returning raw data of frame {self.id} and reasserting frame.data")
            self.data = self.raw_data.copy()
            return self.data
        else:
            print(f"Loading frame {self.id} from disk")
            img = Image.open(self.path)
            if self.index is not None:
                img.seek(self.index)
            self.raw_data = np.asarray(img).astype(np.float)
            self.data = self.raw_data.copy()
            self.width, self.height = self.raw_data.shape
            return self.data

    def clean(self):
        self.translation = [0.0, 0.0]
        self.discard = False
        self.data = None
        self.maxima = list()

    def clone(self):
        return copy.deepcopy(self)

    def bake_transform(self):
        tmat = transform.AffineTransform(np.matrix([[1.0, 0.0, self.translation[0]], [0.0, 1.0, self.translation[1]], [0.0, 0.0, 1.0]]))
        self.data = transform.warp(self.data, tmat)

    def __str__(self):
        selfstr = "Frame at path: "+self.path + "\n" \
            + ("Discarded frame" if self.discard else "Frame in use") \
            + f"\nShift of ({self.translation[0]:.2f}, {self.translation[1]:.2f}) pixels detected."
        if self.maxima is not []:
            selfstr += f"\n{len(self.maxima)} particles found."
        return selfstr


class ParticleData:
    # TODO: make compatible with different kinds of particle datasets, e.g. (x, y, sigma only) or (x, y, z, etc.) - in particular in ParticleData.from_csv

    HISTOGRAM_BINS = 50

    def __init__(self, pixel_size=100):
        self.pixel_size = pixel_size
        self.particles = list()
        self.n_particles = 0
        self.parameter = dict()
        self.histogram_counts = dict()
        self.histogram_bins = dict()
        self.x_max = 0
        self.y_max = 0
        self.x_min = 0
        self.y_min = 0
        self.baked = False  # flag to store whether function bake() has been performed after any changes to the particle list.
        self.baked_by_renderer = False  # flag to store whether renderer has created instance buffers using the latest particle list. ONLY renderer can set it to True, any edits from within ParticleData set it to false.
        self.colours_up_to_date = True
        self.empty = True
        self.reconstruction_roi = [0, 0, 0, 0]

    def __add__(self, other):
        if isinstance(other, list):
            for p in other:
                self.particles.append(p)
            self.baked = False
            self.baked_by_renderer = False
            self.empty = False
            return self

    def __str__(self):
        r = self.reconstruction_roi
        return f"ParticleData obj. with {self.n_particles} particles.\nx range = ({self.x_min}, {self.x_max}), y range = ({self.y_min}, {self.y_max})\nCorresponding ROI in input source image: x {r[0]} to {r[2]}, y {r[1]} to {r[3]}."

    def bake(self):
        self.baked = True
        self.baked_by_renderer = False
        frame = list()
        x = list()
        y = list()
        sigma = list()
        intensity = list()
        offset = list()
        uncertainty = list()
        bkgstd = list()

        self.n_particles = 0
        for p in self.particles:
            self.n_particles += 1
            frame.append(p.frame)
            x.append(p.x)
            y.append(p.y)
            sigma.append(p.sigma)
            intensity.append(p.intensity)
            offset.append(p.offset)
            uncertainty.append(p.uncertainty)
            bkgstd.append(p.bkgstd)

        self.parameter['frame'] = np.asarray(frame)
        self.parameter['x'] = np.asarray(x) * self.pixel_size
        self.parameter['y'] = np.asarray(y) * self.pixel_size
        self.parameter['sigma'] = np.asarray(sigma)
        self.parameter['intensity'] = np.asarray(intensity)
        self.parameter['offset'] = np.asarray(offset)
        self.parameter['uncertainty'] = np.asarray(uncertainty)
        self.parameter['bkgstd'] = np.asarray(bkgstd)

        for key in self.parameter:
            self.histogram_counts[key], self.histogram_bins[key] = np.histogram(self.parameter[key], bins=ParticleData.HISTOGRAM_BINS)
            self.histogram_counts[key] = self.histogram_counts[key].astype(np.float32)
            self.histogram_counts[key] = np.delete(self.histogram_counts[key], 0)
            self.histogram_bins[key] = (self.histogram_bins[key][1], self.histogram_bins[key][-1])

        self.x_min = np.min(self.parameter['x'])
        self.y_min = np.min(self.parameter['y'])
        self.x_max = np.max(self.parameter['x'])
        self.y_max = np.max(self.parameter['y'])

    def set_reconstruction_roi(self, roi):
        """
        :param roi: The ROI used in the initial particle position estimation (e.g. ParticleDetectionNode). It can be saved in ParticleData in order to facilitate overlaying the final reconstruction with the corresponding region of the widefield image.
        :return:
        """
        self.reconstruction_roi = roi

    @staticmethod
    def from_csv(path):
        """
        :param path: path to a .csv file containing super-resolution reconstruction particle data, e.g. from ThunderStorm.
        :return: a ParticleData object.
        """
        df = pd.read_csv(path)
        dfnp = df.to_numpy()
        particles = list()
        for i in range(df.shape[0]):
            _data = df.iloc[i]
            particles.append(Particle(
                _data[0],
                _data[1],
                _data[2],
                _data[3],
                _data[4],
                _data[5],
                _data[6],
                _data[7]
            ))
        particle_data_obj = ParticleData()
        particle_data_obj.particles = particles
        particle_data_obj.bake()
        return particle_data_obj

    def save_as_csv(self, path):
        if not self.baked:
            self.bake()
        for key in self.parameter:
            print(self.parameter[key])
        for key in self.parameter:
            print(key)
            print(self.parameter[key].shape)
        print(self.particles[0])
        pd.DataFrame.from_dict(self.parameter).to_csv(path)


class Particle:

    def __init__(self, frame, x, y, sigma, intensity, offset=0, bkgstd=-1, uncertainty=-1, colour=np.asarray([1.0, 1.0, 1.0]), state=1):
        self.frame = frame
        self.x = x
        self.y = y
        self.sigma = sigma
        self.intensity = intensity
        self.offset = offset
        self.bkgstd = bkgstd
        self.uncertainty = uncertainty
        self.colour = colour
        self.state = state

    def __str__(self):
        return f"{self.frame}, {self.x}, {self.y}, {self.sigma}, {self.intensity}"