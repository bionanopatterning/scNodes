from itertools import count
import numpy as np
import mrcfile
from scNodes.core.opengl_classes import *
import datetime
import scNodes.core.config as cfg
import scNodes.core.settings as settings


class SEFrame:
    idgen = count(0)

    HISTOGRAM_BINS = 40

    def __init__(self, path):
        uid_counter = next(SEFrame.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+"000") + uid_counter
        self.path = path
        self.title = os.path.splitext(os.path.basename(self.path))[0]
        self.n_slices = 0
        self.current_slice = -1
        self.slice_changed = False
        self.data = None
        self.features = list()
        self.active_feature = None
        self.height, self.width = mrcfile.mmap(self.path, mode="r").data.shape[1:3]
        self.pixel_size = mrcfile.open(self.path, header_only=True).voxel_size.x / 10.0
        self.title = f"({self.pixel_size * 10.0:.2f} A/pix) " + self.title
        self.transform = Transform()
        self.texture = None
        self.quad_va = None
        self.border_va = None
        self.interpolate = False
        self.alpha = 1.0
        self.filters = list()
        self.invert = True
        self.autocontrast = True
        self.sample = True
        self.export = False
        self.hist_vals = list()
        self.hist_bins = list()
        self.requires_histogram_update = False
        self.corner_positions_local = []
        self.set_slice(0, False)
        self.setup_opengl_objects()
        self.contrast_lims = [0, 65535.0]
        self.compute_autocontrast()
        self.compute_histogram()

    def setup_opengl_objects(self):
        self.texture = Texture(format="r32f")
        self.texture.update(self.data.astype(np.float32))
        self.quad_va = VertexArray()
        self.border_va = VertexArray(attribute_format="xy")
        self.generate_va()
        self.interpolate = not self.interpolate
        self.toggle_interpolation()

    def toggle_interpolation(self):
        self.interpolate = not self.interpolate
        if self.interpolate:
            self.texture.set_linear_mipmap_interpolation()
        else:
            self.texture.set_no_interpolation()

    def generate_va(self):
        # set up the quad vertex array
        w, h = self.width * 0.5, self.height * 0.5
        vertex_attributes = list()
        indices = list()
        n = cfg.ce_va_subdivision
        for i in range(n):
            for j in range(n):
                x = ((2 * i / (n - 1)) - 1) * w
                y = ((2 * j / (n - 1)) - 1) * h
                u = 0.5 + x / w / 2
                v = 0.5 + y / h / 2
                vertex_attributes += [x, y, 0.0, u, v]

        for i in range(n - 1):
            for j in range(n - 1):
                idx = i * n + j
                indices += [idx, idx + 1, idx + n, idx + n, idx + 1, idx + n + 1]

        self.corner_positions_local = [[-w, h], [-w, -h], [w, -h], [w, h]]
        self.quad_va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))

        # set up the border vertex array
        vertex_attributes = [-w, h,
                             w, h,
                             w, -h,
                             -w, -h]
        indices = [0, 1, 1, 2, 2, 3, 3, 0]
        self.border_va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))

    def set_slice(self, requested_slice, update_texture=True):
        self.requires_histogram_update = True
        if requested_slice == self.current_slice:
            return
        self.slice_changed = True
        mrc = mrcfile.mmap(self.path, mode="r")
        self.n_slices = mrc.data.shape[0]
        requested_slice = min([max([requested_slice, 0]), self.n_slices - 1])
        self.data = mrc.data[requested_slice, :, :]
        target_type_dict = {np.float32: float, float: float, np.int8: np.uint8, np.int16: np.uint16}
        if type(self.data[0, 0]) not in target_type_dict:
            target_type = float
        else:
            target_type = target_type_dict[type(self.data[0, 0])]
        self.data = np.array(self.data.astype(target_type, copy=False), dtype=float)
        self.current_slice = requested_slice
        for s in self.features:
            s.set_slice(self.current_slice)
        if update_texture:
            self.update_image_texture()

    def update_image_texture(self):
        self.texture.update(self.data.astype(np.float32))

    def update_model_matrix(self):
        self.transform.scale = self.pixel_size
        self.transform.compute_matrix()
        for i in range(4):
            vec = np.matrix([*self.corner_positions_local[i], 0.0, 1.0]).T
            corner_pos = (self.transform.matrix * vec)[0:2]
            self.corner_positions_local[i] = [float(corner_pos[0]), float(corner_pos[1])]

    def world_to_pixel_coordinate(self, world_coordinate):
        vec = np.matrix([world_coordinate[0], world_coordinate[1], 0.0, 1.0]).T
        invmat = np.linalg.inv(self.transform.matrix)
        out_vec = invmat * vec
        pixel_coordinate = [int(out_vec[0, 0] + self.width / 2), int(out_vec[1, 0] + self.height / 2)]
        return pixel_coordinate

    def compute_autocontrast(self, saturation=None, pxd=None):
        data = self.data if pxd is None else pxd
        #s = data.shape
        #data = data[int(s[0]/3):int(s[0]*2/3), int(s[1]/3):int(s[1]*2/3)]
        saturation_pct = settings.autocontrast_saturation
        if saturation:
            saturation_pct = saturation
        subsample = data[::settings.autocontrast_subsample, ::settings.autocontrast_subsample]
        n = subsample.shape[0] * subsample.shape[1]
        sorted_pixelvals = np.sort(subsample.flatten())

        min_idx = min([int(saturation_pct / 100.0 * n), n - 1])
        max_idx = max([int((1.0 - saturation_pct / 100.0) * n), 0])
        self.contrast_lims[0] = sorted_pixelvals[min_idx]
        self.contrast_lims[1] = sorted_pixelvals[max_idx]

    def compute_histogram(self, pxd=None):
        self.requires_histogram_update = False
        data = self.data if pxd is None else pxd
        # ignore very bright pixels
        mean = np.mean(data)
        std = np.std(data)
        self.hist_vals, self.hist_bins = np.histogram(data[data < (mean + 20 * std)], bins=SEFrame.HISTOGRAM_BINS)

        self.hist_vals = self.hist_vals.astype('float32')
        self.hist_bins = self.hist_bins.astype('float32')
        self.hist_vals = np.log(self.hist_vals + 1)

    def on_load(self):
        uid_counter = next(SEFrame.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        self.setup_opengl_objects()
        for f in self.features:
            f.on_load()
        for f in self.filters:
            f.fill_kernel()




    def __eq__(self, other):
        if isinstance(other, SEFrame):
            return self.uid == other.uid
        return False


class Filter:

    TYPES = ["Gaussian blur", "Box blur", "Sobel vertical", "Sobel horizontal"]
    PARAMETER_NAME = ["sigma", "box", None, None]
    M = 16

    def __init__(self, parent, filter_type):
        self.parent = parent
        self.type = filter_type  # integer, corresponding to an index in the Filter.TYPES list
        self.k1 = np.zeros((Filter.M*2+1, 1), dtype=np.float32)
        self.k2 = np.zeros((Filter.M * 2 + 1, 1), dtype=np.float32)
        self.enabled = True
        self.ssbo1 = -1
        self.ssbo2 = -1
        self.param = 1.0
        self.strength = 1.0
        self.fill_kernel()

    def upload_buffer(self):
        if self.ssbo1 != -1:
            glDeleteBuffers(2, [self.ssbo1, self.ssbo2])
        self.ssbo1 = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo1)
        glBufferData(GL_SHADER_STORAGE_BUFFER, (Filter.M * 2 + 1) * 4, self.k1.flatten(), GL_STATIC_READ)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        self.ssbo2 = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo2)
        glBufferData(GL_SHADER_STORAGE_BUFFER, (Filter.M * 2 + 1) * 4, self.k2.flatten(), GL_STATIC_READ)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)


    def bind(self, horizontal=True):
        if horizontal:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo1)
        else:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo2)

    def unbind(self):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def fill_kernel(self):
        if self.type == 0:
            self.k1 = np.exp(-np.linspace(-Filter.M, Filter.M, 2*Filter.M+1)**2 / self.param**2, dtype=np.float32)
            self.k1 /= np.sum(self.k1, dtype=np.float32)
            self.k2 = self.k1
        if self.type == 1:
            m = min([int(self.param), 7])
            self.k1 = np.asarray([0] * (Filter.M - 1 - m) + [1] * m + [1] + [1] * m + [0] * (Filter.M - 1 - m)) / (2 * m + 1)
            self.k2 = self.k1
        if self.type == 2:
            self.k1 = np.asarray([0] * (Filter.M - 1) + [1, 2, 1] + [0] * (Filter.M - 1))
            self.k2 = np.asarray([0] * (Filter.M - 1) + [1, 0, -1] + [0] * (Filter.M - 1))
        if self.type == 3:
            self.k1 = np.asarray([0] * (Filter.M - 1) + [1, 0, -1] + [0] * (Filter.M - 1))
            self.k2 = np.asarray([0] * (Filter.M - 1) + [1, 2, 1] + [0] * (Filter.M - 1))

        self.k1 = np.asarray(self.k1, dtype=np.float32)
        self.k2 = np.asarray(self.k2, dtype=np.float32)
        self.upload_buffer()

        self.parent.requires_histogram_update = True


class Segmentation:
    idgen = count(0)

    DEFAULT_COLOURS = [(66 / 255, 214 / 255, 164 / 255),
                       (255 / 255, 243 / 255, 0 / 255),
                       (255 / 255, 104 / 255, 0 / 255),
                       (255 / 255, 13 / 255, 0 / 255),
                       (174 / 255, 0 / 255, 255 / 255),
                       (21 / 255, 0 / 255, 255 / 255),
                       (0 / 255, 136 / 255, 266 / 255),
                       (0 / 255, 247 / 255, 255 / 255),
                       (0 / 255, 255 / 255, 0 / 255)]


    def __init__(self, parent_frame, title):
        uid_counter = next(Segmentation.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        self.parent = parent_frame
        self.parent.active_feature = self
        self.width = self.parent.width
        self.height = self.parent.height
        self.title = title
        self.colour = Segmentation.DEFAULT_COLOURS[(len(self.parent.features)) % len(Segmentation.DEFAULT_COLOURS)]
        self.alpha = 0.33
        self.hide = False
        self.contour = False
        self.expanded = False
        self.brush_size = 10.0
        self.show_boxes = True
        self.box_size = 32
        self.box_size_nm = self.box_size * self.parent.pixel_size
        self.slices = dict()
        self.boxes = dict()
        self.n_boxes = 0
        self.edited_slices = list()
        self.current_slice = -1
        self.data = None
        self.texture = Texture(format="r32f")
        self.texture.update(None, self.width, self.height)
        self.texture.set_linear_interpolation()
        self.set_slice(self.parent.current_slice)

    def on_load(self):
        uid_counter = next(Segmentation.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        self.texture = Texture(format="r32f")
        cslice = self.current_slice
        self.current_slice = -1
        self.set_slice(cslice)

    def set_box_size(self, box_size_px):
        self.box_size = box_size_px
        self.box_size_nm = self.box_size * self.parent.pixel_size

    def add_box(self, pixel_coordinates):
        if self.current_slice not in self.boxes:
            self.boxes[self.current_slice] = list()
        self.request_draw_in_current_slice()
        self.boxes[self.current_slice].append(pixel_coordinates)
        self.n_boxes += 1

    def remove_box(self, pixel_coordinate):
        box_list = self.boxes[self.current_slice]
        x = pixel_coordinate[0]
        y = pixel_coordinate[1]
        d = np.inf
        idx = None
        for i in range(len(box_list)):
            _d = (box_list[i][0]-x)**2 + (box_list[i][1]-y)**2
            if _d < d:
                d = _d
                idx = i
        if idx is not None:
            self.boxes[self.current_slice].pop(idx)
            self.n_boxes -= 1

    def set_slice(self, requested_slice):
        if requested_slice == self.current_slice:
            return
        self.current_slice = requested_slice
        if requested_slice in self.slices:
            self.data = self.slices[requested_slice]
            if self.data is None:
                self.texture.update(self.data, self.width, self.height)
            else:
                self.texture.update(self.data)
        else:
            self.slices[requested_slice] = None
            if requested_slice not in self.boxes:
                self.boxes[requested_slice] = list()
            self.data = None
            self.texture.update(self.data, self.width, self.height)

    def remove_slice(self, requested_slice):
        if requested_slice in self.edited_slices:
            self.edited_slices.remove(requested_slice)
            self.slices.pop(requested_slice)
            if self.current_slice == requested_slice:
                self.data *= 0
                self.texture.update(self.data, self.width, self.height)
        if requested_slice in self.boxes:
            self.n_boxes -= len(self.boxes[requested_slice])
            self.boxes[requested_slice] = list()

    def request_draw_in_current_slice(self):
        if self.current_slice in self.slices:
            if self.slices[self.current_slice] is None:
                self.slices[self.current_slice] = np.zeros((self.height, self.width), dtype=np.uint8)
                self.data = self.slices[self.current_slice]
                self.texture.update(self.data, self.width, self.height)
                self.edited_slices.append(self.current_slice)
        else:
            self.slices[self.current_slice] = np.zeros((self.height, self.width), dtype=np.uint8)
            self.data = self.slices[self.current_slice]
            self.texture.update(self.data, self.width, self.height)
            self.edited_slices.append(self.current_slice)


class Transform:
    def __init__(self):
        self.translation = np.array([0.0, 0.0])
        self.rotation = 0.0
        self.scale = 1.0
        self.matrix = np.identity(4)
        self.matrix_no_scale = np.identity(4)

    def compute_matrix(self):
        scale_mat = np.identity(4) * self.scale
        scale_mat[3, 3] = 1

        rotation_mat = np.identity(4)
        _cos = np.cos(self.rotation / 180.0 * np.pi)
        _sin = np.sin(self.rotation / 180.0 * np.pi)
        rotation_mat[0, 0] = _cos
        rotation_mat[1, 0] = _sin
        rotation_mat[0, 1] = -_sin
        rotation_mat[1, 1] = _cos

        translation_mat = np.identity(4)
        translation_mat[0, 3] = self.translation[0]
        translation_mat[1, 3] = self.translation[1]

        self.matrix = np.matmul(translation_mat, np.matmul(rotation_mat, scale_mat))
        self.matrix_no_scale = np.matmul(translation_mat, rotation_mat)

    def __add__(self, other):
        out = Transform()
        out.translation[0] = self.translation[0] + other.translation[0]
        out.translation[1] = self.translation[1] + other.translation[1]
        out.rotation = self.rotation + other.rotation
        return out

    def __str__(self):
        return f"Transform with translation = {self.translation[0], self.translation[1]}, scale = {self.scale}, rotation = {self.rotation}"

    def __sub__(self, other):
        out = Transform()
        out.translation[0] = self.translation[0] - other.translation[0]
        out.translation[1] = self.translation[1] - other.translation[1]
        out.rotation = self.rotation - other.rotation
        return out
