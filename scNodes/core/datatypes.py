from itertools import count
import numpy as np
import re
import glob
from PIL import Image
from skimage import transform
import pandas as pd
import copy
import tifffile


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

        img = Image.open(self.path)
        tif = tifffile.TiffFile(self.path)
        n_frames = len(tif.pages)
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
        if index is None:
            index = self.current_frame
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
        original_active_frame = self.frames[self.current_frame]
        discard = [False] * len(self.frames)
        neg_tags = negative_filter_string.split(';')
        pos_tags = positive_filter_string.split(';')
        while '' in neg_tags:
            neg_tags.remove('')
        while '' in pos_tags:
            pos_tags.remove('')
        i = 0
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
        if original_active_frame in self.frames:
            self.current_frame = self.frames.index(original_active_frame)

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

    def append_dataset(self, path):
        new_dataset = Dataset(path, self.pixel_size)
        if not self.initialized or not new_dataset.initialized:
            return
        if self.img_width != new_dataset.img_width:
            print("Original dataset and dataset to append are not the same size!")
            return
        if self.img_height != new_dataset.img_height:
            print("Original dataset and dataset to append are not the same size!")
            return
        for f in new_dataset.frames:
            self.frames.append(f)
            self.n_frames += 1


    def delete_by_index(self, idx):
        if 0 <= idx < len(self.frames):
            del self.frames[idx]
            self.n_frames = len(self.frames)

    def __str__(self):
        return "Dataset object with source at: "+self.path+f"\nn_frames = {self.n_frames}"


class Frame:
    id_gen = count(1)

    def __init__(self, path, index=None, framenr=0):
        self.id = next(Frame.id_gen)
        self.index = index
        self.framenr = framenr
        self.path = path
        if "/" in self.path:
            self.title = self.path[self.path[:-1].rfind("/"):]
        else:
            self.title = self.path
        self.data = None
        self.raw_data = None
        self.width = None
        self.height = None
        self.discard = False
        self.translation = [0.0, 0.0]
        self.maxima = list()
        self.maxima_values = list()
        self.particles = list()
        self.pixel_size = 100
        self.gpu_data_buffer = None
        self.gpu_params_buffer = None
        self.gpu_crop_xy_buffer = None
        self._ce_lut = 0
        self._ce_clims = [0, 1]
        self.scalar_metrics = dict()

    def load(self):
        if self.data is not None:
            return self.data
        else:
            if self.index is None:
                self.data = tifffile.imread(self.path).astype(float)
            else:
                self.data = tifffile.imread(self.path, key=self.index).astype(float)
            self.width = self.data.shape[0]
            self.height = self.data.shape[1]
            return self.data

    def clean(self):
        self.translation = [0.0, 0.0]
        self.discard = False
        self.data = None
        self.maxima = list()

    def clone(self):
        return copy.deepcopy(self)

    def bake_transform(self, interpolation=1, edges='constant', preserve_range=False):
        tmat = transform.AffineTransform(np.matrix([[1.0, 0.0, self.translation[0]], [0.0, 1.0, self.translation[1]], [0.0, 0.0, 1.0]]))
        self.data = transform.warp(self.data, tmat, order=interpolation, mode=edges, preserve_range=preserve_range)

    def load_roi(self, roi=None):
        if roi is None:
            return self.load()
        else:
            return self.load()[roi[1]:roi[3], roi[0]:roi[2]]

    def write_roi(self, roi, data):
        self.data[roi[1]:roi[3], roi[0]:roi[2]] = data

    def __str__(self):
        sstr = "Frame: ..." + self.title + "\n" \
            + ("Discarded frame. " if self.discard else "Frame in use. ") \
            + f"\nShift of ({self.translation[0]:.2f}, {self.translation[1]:.2f}) pixels detected."
        if self.maxima is not []:
            sstr += f"\n{len(self.maxima)} particles found."
        return sstr

    def __eq__(self, other):
        if isinstance(other, Frame):
            return self.path == other.path and self.framenr == other.framenr
        return False

class ParticleData:
    HISTOGRAM_BINS = 50

    def __init__(self, pixel_size=100):
        self.pixel_size = pixel_size
        self.parameters = dict()
        self.n_particles = 0
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
        self.photons_per_count = 0.45
        self.camera_offset = 100.0
        self.uncertainty_estimator = 0

    def __add__(self, other):
        if not other:
            return self
        for key in other:
            if key in self.parameters:
                self.parameters[key] += other[key]
            else:
                self.parameters[key] = other[key]
        self.baked = False
        self.baked_by_renderer = False
        self.empty = False
        self.n_particles = len(next(iter(self.parameters.values())))
        return self

    def __str__(self):
        r = self.reconstruction_roi
        return f"ParticleData obj. with {self.n_particles} particles.\nx range = ({self.x_min}, {self.x_max}), y range = ({self.y_min}, {self.y_max})\nCorresponding ROI in input source image: x {r[0]} to {r[2]}, y {r[1]} to {r[3]}."

    def clean(self):
        if 'visible' in self.parameters:
            self.parameters["visible"] = np.ones_like(self.parameters["visible"])
        if "dx [nm]" in self.parameters:
            self.parameters["dx [nm]"] = np.zeros_like(self.parameters["dx [nm]"])
        if "dy [nm]" in self.parameters:
            self.parameters["dy [nm]"] = np.zeros_like(self.parameters["dy [nm]"])
        self.baked_by_renderer = False

    def bake(self):
        if self.empty:
            return
        self.baked = True
        self.baked_by_renderer = False

        for key in self.parameters:
            self.parameters[key] = np.array([v for a in self.parameters[key] for v in (a if isinstance(a, list) else [a])])

        self.n_particles = len(self.parameters["x [nm]"])
        print("n_particles:", self.n_particles)

        if self.uncertainty_estimator == 0:
            bkgstd = self.parameters["bkgstd [counts]"]
            intensity = self.parameters["intensity [counts]"]
            k1 = self.pixel_size ** 2 / 12
            k2 = 8 * np.pi / self.pixel_size ** 2
        elif self.uncertainty_estimator == 1:
            self.parameters["intensity [photons]"] = self.parameters["intensity [counts]"] * self.photons_per_count
            self.parameters["offset [photons]"] = self.parameters["offset [counts]"] * self.photons_per_count
            # self.parameters.pop("intensity [counts]")
            # self.parameters.pop("offset [counts]")
            intensity = self.parameters["intensity [photons]"]
            background = self.parameters["offset [photons]"]

        sigma = self.parameters["sigma [nm]"]
        discard_mask = np.zeros_like(self.parameters["x [nm]"])
        uncertainty = np.zeros_like(self.parameters["x [nm]"])

        for i in range(self.n_particles):
            if intensity[i] == 0.0:
                discard_mask[i] = 1
                continue
            if self.uncertainty_estimator == 0:
                uncertainty[i] = np.sqrt(((sigma[i]*self.pixel_size)**2 + k1) / intensity[i] + k2 * (sigma[i]*self.pixel_size)**4 * bkgstd[i]**2 / intensity[i]**2)
            elif self.uncertainty_estimator == 1:
                N = intensity[i]
                sa2 = (sigma[i]**2 + 1 / 12) * self.pixel_size**2
                tau = (2 * np.pi * (sigma[i]**2 + 1/12) * background[i]) / N
                variance = sa2 * (1 + 4*tau + np.sqrt((2 * tau)/(1 + 4*tau))) / N
                uncertainty[i] = np.sqrt(variance)

        discard_mask += self.parameters["x [nm]"] < self.x_min
        discard_mask += self.parameters["x [nm]"] > self.x_max
        discard_mask += self.parameters["y [nm]"] < self.y_min
        discard_mask += self.parameters["y [nm]"] > self.y_max

        discard_indices = discard_mask.nonzero()
        self.parameters['uncertainty [nm]'] = np.asarray(uncertainty)

        self.parameters['x [nm]'] *= self.pixel_size
        self.parameters['y [nm]'] *= self.pixel_size
        self.parameters['sigma [nm]'] *= self.pixel_size
        if 'sigma2 [nm]' in self.parameters.keys():
            self.parameters['sigma2 [nm]'] *= self.pixel_size

        for key in self.parameters:
            np.delete(self.parameters[key], discard_indices)

        self.parameters['visible'] = np.ones_like(self.parameters['x [nm]'])
        self.parameters['colour_idx'] = np.ones_like(self.parameters['x [nm]'])
        for key in self.parameters:
            self.histogram_counts[key], self.histogram_bins[key] = np.histogram(self.parameters[key], bins=ParticleData.HISTOGRAM_BINS)
            self.histogram_counts[key] = self.histogram_counts[key].astype(np.float32)
            self.histogram_counts[key] = np.delete(self.histogram_counts[key], 0)
            self.histogram_bins[key] = (self.histogram_bins[key][1], self.histogram_bins[key][-1])

        self.x_min = np.min(self.parameters['x [nm]'])
        self.y_min = np.min(self.parameters['y [nm]'])
        self.x_max = np.max(self.parameters['x [nm]'])
        self.y_max = np.max(self.parameters['y [nm]'])
        self.n_particles = len(self.parameters["x [nm]"])
        self.parameters['dx [nm]'] = np.zeros_like(self.parameters['x [nm]'])
        self.parameters['dy [nm]'] = np.zeros_like(self.parameters['y [nm]'])

    def apply_filter(self, parameter_key, min_val, max_val, logic_not=False):
        if parameter_key not in self.parameters:
            return
        vals = self.parameters[parameter_key]
        visible = np.zeros_like(self.parameters['visible'])
        for i in range(self.n_particles):
            if min_val < vals[i] < max_val:
                visible[i] = 1
        if logic_not:
            visible = 1 - visible
        self.parameters['visible'][visible == 0] = 0

    def set_reconstruction_roi(self, roi):
        """
        :param roi: The ROI used in the initial particle position estimation (e.g. ParticleDetectionNode). It can be saved in ParticleData in order to facilitate overlaying the final reconstruction with the corresponding region of the widefield image.
        """
        self.reconstruction_roi = roi

    @staticmethod
    def from_csv(path):
        """
        :param path: path to a .csv file containing super-resolution reconstruction particle data, e.g. from ThunderStorm.
        :return: a ParticleData object.
        """
        df = pd.read_csv(path)
        particle_data_obj = ParticleData()
        for key in df.keys():
            particle_data_obj.parameters[key] = df.values[:, df.columns.get_loc(key)]

        particle_data_obj.pixel_size = 1
        particle_data_obj.n_particles = len(particle_data_obj.parameters['x [nm]'])
        particle_data_obj.parameters['dx [nm]'] = np.ones_like(particle_data_obj.parameters['x [nm]'])
        particle_data_obj.parameters['dy [nm]'] = np.ones_like(particle_data_obj.parameters['x [nm]'])
        particle_data_obj.parameters['visible'] = np.ones_like(particle_data_obj.parameters['x [nm]'])
        particle_data_obj.parameters['colour_idx'] = np.ones_like(particle_data_obj.parameters['x [nm]'])
        particle_data_obj.x_min = np.min(particle_data_obj.parameters['x [nm]'])
        particle_data_obj.y_min = np.min(particle_data_obj.parameters['y [nm]'])
        particle_data_obj.x_max = np.max(particle_data_obj.parameters['x [nm]'])
        particle_data_obj.y_max = np.max(particle_data_obj.parameters['y [nm]'])
        particle_data_obj.empty = False
        particle_data_obj.set_reconstruction_roi([particle_data_obj.x_min, particle_data_obj.y_min, particle_data_obj.x_max, particle_data_obj.y_max])

        for key in particle_data_obj.parameters:
            particle_data_obj.histogram_counts[key], particle_data_obj.histogram_bins[key] = np.histogram(particle_data_obj.parameters[key], bins=ParticleData.HISTOGRAM_BINS)
            particle_data_obj.histogram_counts[key] = particle_data_obj.histogram_counts[key].astype(np.float32)
            particle_data_obj.histogram_counts[key] = np.delete(particle_data_obj.histogram_counts[key], 0)
            particle_data_obj.histogram_bins[key] = (particle_data_obj.histogram_bins[key][1], particle_data_obj.histogram_bins[key][-1])
        particle_data_obj.baked = True
        return particle_data_obj

    def save_as_csv(self, path):
        _cidx = False
        if "colour_idx" in self.parameters:
            _colour_idx = self.parameters.pop("colour_idx")
            _cidx = True
        _v = False
        if "visible" in self.parameters:
            _visible = self.parameters.pop("visible")
            _v = True

        _dx = False
        _dy = False
        if "dx [nm]" in self.parameters:
            dx = self.parameters.pop("dx [nm]")
            _dx = True
        if "dy [nm]" in self.parameters:
            dy = self.parameters.pop("dy [nm]")
            _dy = True
        self.parameters["x [nm]"] += dx
        self.parameters["y [nm]"] += dy
        pd.DataFrame.from_dict(self.parameters).to_csv(path, index=False)
        if _cidx:
            self.parameters["colour_idx"] = _colour_idx
        if _v:
            self.parameters["visible"] = _visible
        if _dx:
            self.parameters["dx [nm]"] = dx
        if _dy:
            self.parameters["dy [nm]"] = dy
