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
        """
        :param path: None or string, path to either i) a multi-page .tif file with ordering XYF, F the number of frames, or ii) a single-page .tif file, in which case a Dataset is generated comprising all the .tif files in that folder. If 'None', a Dataset() object is generated that has no frame data.
        :param pixel_size:
        """
        self.id = next(Dataset.idgen)
        self.path = path
        self.frames = list()
        self.n_frames = 0
        self.current_frame = 0
        self.pixel_size = pixel_size
        self.initialized = False
        self.reconstruction_roi = [0, 0, 0, 0]
        if self.path is not None:
            self.directory = self.path[:self.path.rfind("/") + 1]
            self.load_data()
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
                self.frames.append(Frame(self.path, i, framenr=i))

        # folder
        else:
            files = sorted(glob.glob(self.directory + "*.tif*"), key=numerical_sort)
            for file in files:
                self.n_frames += 1
                self.frames.append(Frame(file, 0, framenr=self.n_frames - 1))

    def get_indexed_image(self, index):
        if 0 <= index < self.n_frames:
            self.frames[index].pixel_size = self.pixel_size
            return self.frames[index]
        else:
            if index <= 0:
                self.frames[0].pixel_size = self.pixel_size
                return self.frames[0]
            else:
                self.frames[-1].pixel_size = self.pixel_size
                return self.frames[-1]

    def get_active_image(self):
        self.frames[self.current_frame].pixel_size = self.pixel_size
        return self.frames[self.current_frame]

    def filter_frames_by_title(self, positive_filter_string, negative_filter_string):
        """
        :param positive_filter_string: string, all frames not containing any of the semicolon separated expressions in this string are dicarded
        :param negative_filter_string: string, all frames containing any of the semicolon separated expresions in this string are discarded.
        Note that when positive and negative tags contradict, preference is given to the positive filter. I.e., the frame is not discarded.
        :return: Nothing, dataset object itself is affected
        """
        print("Filtering")
        discard = [False] * len(self.frames)
        neg_tags = negative_filter_string.split(';')
        pos_tags = positive_filter_string.split(';')
        while '' in neg_tags:
            neg_tags.remove('')
        while '' in pos_tags:
            pos_tags.remove('')
        i = 0
        print(neg_tags)
        print(pos_tags)
        for frame in self.frames:
            for tag in neg_tags:
                if tag in frame.path:
                    discard[i] = True
            for tag in pos_tags:
                if tag in frame.path:
                    discard[i] = False
                else:
                    discard[i] = True
            i += 1
        for i in range(len(self.frames) - 1, -1, -1):
            if discard[i]:

                self.frames.pop(i)
        self.n_frames = len(self.frames)

    def append_frame(self, frame):
        """
        Add a frame to the end of the list of frames in the dataset.
        :param frame: Frame object (see below).
        """
        if not self.initialized:
            self.img_width, self.img_height = frame.load().shape
            self.initialized = True
        self.frames.append(frame)
        self.n_frames += 1

    def __str__(self):
        return "Dataset object with source at: "+self.path+f"\nn_frames = {self.n_frames}"


class Frame:
    id_gen = count(1)

    def __init__(self, path, index=None, framenr=0):
        self.id = next(Frame.id_gen)
        self.index = index
        self.framenr = framenr
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
        self.gpu_data_buffer = None
        self.gpu_params_buffer = None
        self.gpu_crop_xy_buffer = None

    def load(self):
        if self.data is not None:
            return self.data
        elif self.raw_data is not None:
            self.data = self.raw_data.copy()
            return self.data
        else:
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
        self.data = None ## TODO: check whether app is faster and output still correct when this line is removed.
        self.maxima = list()

    def clone(self):
        return copy.deepcopy(self)

    def bake_transform(self, interpolation = 1, edges = 'constant', preserve_range=False):
        tmat = transform.AffineTransform(np.matrix([[1.0, 0.0, self.translation[0]], [0.0, 1.0, self.translation[1]], [0.0, 0.0, 1.0]]))
        self.data = transform.warp(self.data, tmat, order=interpolation, mode=edges, preserve_range=preserve_range)

    def __str__(self):
        sstr = f"Frame at path: {self.path}\n" \
            + ("Discarded frame. " if self.discard else "Frame in use. ") \
            + f"Shift of ({self.translation[0]:.2f}, {self.translation[1]:.2f}) pixels detected."
        if self.maxima is not []:
            sstr += f"\n{len(self.maxima)} particles found."
        return sstr


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

    def clean(self):
        for particle in self.particles:
            particle.visible = True
        self.baked_by_renderer = False

    def bake(self, discard_filtered_particles=False):
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

        k1 = self.pixel_size**2 / 12
        k2 = 8 * np.pi / self.pixel_size**2
        self.n_particles = 0
        for p in self.particles:
            if p.visible or not discard_filtered_particles:
                self.n_particles += 1
                frame.append(p.frame)
                x.append(p.x)
                y.append(p.y)
                sigma.append(p.sigma * self.pixel_size)
                intensity.append(p.intensity)
                offset.append(p.offset)
                p.uncertainty = np.sqrt(((p.sigma*self.pixel_size)**2 + k1) / p.intensity + k2 * (p.sigma*self.pixel_size)**4 * p.bkgstd**2 / p.intensity**2)
                uncertainty.append(p.uncertainty)
                bkgstd.append(p.bkgstd)


        self.parameter['uncertainty [nm]'] = np.asarray(uncertainty)
        self.parameter['intensity [counts]'] = np.asarray(intensity)
        self.parameter['offset [counts]'] = np.asarray(offset)
        self.parameter['x [nm]'] = np.asarray(x) * self.pixel_size
        self.parameter['y [nm]'] = np.asarray(y) * self.pixel_size
        self.parameter['sigma [nm]'] = np.asarray(sigma)
        self.parameter['bkgstd [counts]'] = np.asarray(bkgstd)
        self.parameter['frame'] = np.asarray(frame).astype(np.float32)

        for key in self.parameter:
            self.histogram_counts[key], self.histogram_bins[key] = np.histogram(self.parameter[key], bins=ParticleData.HISTOGRAM_BINS)
            self.histogram_counts[key] = self.histogram_counts[key].astype(np.float32)
            self.histogram_counts[key] = np.delete(self.histogram_counts[key], 0)
            self.histogram_bins[key] = (self.histogram_bins[key][1], self.histogram_bins[key][-1])

        self.x_min = np.min(self.parameter['x [nm]'])
        self.y_min = np.min(self.parameter['y [nm]'])
        self.x_max = np.max(self.parameter['x [nm]'])
        self.y_max = np.max(self.parameter['y [nm]'])

    def apply_filter(self, parameter_key, min_val, max_val, logic_not = False):
        vals = self.parameter[parameter_key]
        if logic_not:
            for p, v in zip(self.particles, vals):
                if min_val < v < max_val:
                    p.visible = False
        else:
            for p, v in zip(self.particles, vals):
                if v < min_val or v > max_val:
                    p.visible = False



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
        pd.DataFrame.from_dict(self.parameter).to_csv(path, index=False)


class Particle:

    def __init__(self, frame, x, y, sigma, intensity, offset=0, bkgstd=-1, uncertainty=-1, colour=np.asarray([1.0, 1.0, 1.0])):
        self.frame = frame
        self.x = x
        self.y = y
        self.sigma = sigma
        self.intensity = intensity
        self.offset = offset
        self.bkgstd = bkgstd
        self.uncertainty = uncertainty
        self.colour = colour
        self.visible = True

    def __str__(self):
        return f"f={self.frame}, x={self.x}, y={self.y}, sigma={self.sigma}, i={self.intensity}, offset={self.offset}, bkgstd={self.bkgstd}"
