
import shutil
from itertools import count
import imgui
from imgui.integrations.glfw import GlfwRenderer
import config as cfg
import glfw
import time
import datetime
import util
from opengl_classes import *
import tkinter as tk
from tkinter import filedialog
from dataset import *
from reconstruction import *
from util import *
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.signal import medfilt
from skimage.feature import peak_local_max
from pystackreg import StackReg
import pywt
from joblib import Parallel, delayed, cpu_count
import particlefitting as pfit
import dill as pickle
import copy
import psutil
import cv2
tkroot = tk.Tk()
tkroot.withdraw()

class LoadDataNode(Node):

    def __init__(self):
        super().__init__(Node.TYPE_LOAD_DATA) #Was: super(LoadDataNode, self).__init__()
        self.size = [200, 200]

        # Set up connectable attributes
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)
        self.connectable_attributes.append(self.dataset_out)
        # Set up node-specific vars
        self.dataset = Dataset()
        self.path = ""
        self.pixel_size = 64.0
        self.load_on_the_fly = True
        self.done_loading = False
        self.to_load_idx = 0
        self.n_to_load = 1

        self.file_filter_positive_raw = ""
        self.file_filter_negative_raw = ""

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_out.render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.text("Select source file")
            imgui.push_item_width(150)
            _, self.path = imgui.input_text("##intxt", self.path, 256, imgui.INPUT_TEXT_ALWAYS_OVERWRITE)
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("...", 26, 19):
                selected_file = filedialog.askopenfilename()
                if selected_file is not None:
                    if get_filetype(selected_file) in ['.tiff', '.tif']:
                        self.path = selected_file
                        self.on_select_file()
            imgui.columns(2, border = False)
            imgui.text("frames:")
            imgui.text("image size:")
            imgui.text("pixel size:  ")
            imgui.next_column()
            imgui.new_line()
            imgui.same_line(spacing=3)
            imgui.text(f"{self.dataset.n_frames}")
            imgui.new_line()
            imgui.same_line(spacing=3)
            imgui.text(f"{self.dataset.img_width}x{self.dataset.img_height}")
            imgui.push_item_width(45)
            _c, self.pixel_size = imgui.input_float("##nm", self.pixel_size, 0.0, 0.0, format = "%.1f")
            self.any_change = self.any_change or _c
            if _c:
                self.dataset.pixel_size = self.pixel_size
            imgui.pop_item_width()
            imgui.same_line()
            imgui.text("nm")
            imgui.columns(1)


            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()

    def render_advanced(self):
        _, self.load_on_the_fly = imgui.checkbox("Load on the fly", self.load_on_the_fly)
        if not self.load_on_the_fly and not self.done_loading:
            self.progress_bar(min([self.to_load_idx / self.n_to_load]))
            imgui.spacing()
            imgui.spacing()
            imgui.spacing()

        imgui.text("Title must contain:")
        av_width = imgui.get_content_region_available_width()
        imgui.set_next_item_width(av_width)
        _c, self.file_filter_positive_raw = imgui.input_text("##filt_pos", self.file_filter_positive_raw, 1024, imgui.INPUT_TEXT_AUTO_SELECT_ALL)
        Node.tooltip("Enter tags, e.g. 'GFP;RFP', that must be in a filename\n"
                     "in order to retain the frame. Separate tags by a semicolon.\n"
                     "When multiple tags are entered, frames are retained if \n"
                     "any of the tags is in the filename (not necessarily all).\n"
                     "Leave empty to retain all files by default.")
        imgui.text("Title must not contain:")
        imgui.set_next_item_width(av_width)
        _c, self.file_filter_negative_raw = imgui.input_text("##filt_neg", self.file_filter_negative_raw, 1024, imgui.INPUT_TEXT_AUTO_SELECT_ALL)
        Node.tooltip("Enter tags, e.g. 'GFP;RFP', to select frames for deletion.\n"
                     "Separate by a semicolon. When the positive and negative\n"
                     "selection criteria contradict, frames are retained.")
        if imgui.button("Filter", av_width / 2 - 5, 25):
            self.dataset.filter_frames_by_title(self.file_filter_positive_raw, self.file_filter_negative_raw)
            self.any_change = True
        imgui.same_line(spacing=10)
        if imgui.button("Reset", av_width / 2 - 5, 25):
            self.on_select_file()
            self.any_change = True

    def on_select_file(self):
        self.dataset = Dataset(self.path, self.pixel_size)
        self.n_to_load = self.dataset.n_frames
        self.done_loading = False
        self.to_load_idx = 0
        self.any_change = True
        cfg.image_viewer.center_image_requested = True
        NodeEditor.set_active_node(self)

    def get_image_impl(self, idx):
        if self.dataset.n_frames > 0:
            retimg = copy.deepcopy(self.dataset.get_indexed_image(idx))
            retimg.clean()
            return retimg
        else:
            return None

    def on_update(self):
        if not self.load_on_the_fly and not self.done_loading:
            if NodeEditor.profiling:
                time_start = time.time()
                self.profiler_count += 1
            if self.to_load_idx < self.dataset.n_frames:
                self.dataset.get_indexed_image(self.to_load_idx).load()
                self.to_load_idx += 1
            else:
                self.done_loading = True
            if NodeEditor.profiling:
                self.profiler_time += time.time() - time_start

    def pre_save_impl(self):
        NodeEditor.pickle_temp["dataset"] = self.dataset
        self.dataset = Dataset()

    def post_save_impl(self):
        self.dataset = NodeEditor.pickle_temp["dataset"]


class RegisterNode(Node):
    METHODS = ["TurboReg", "ORB"]
    REFERENCES = ["Input image", "Template frame", "Consecutive pairing"]
    INTERPOLATION_OPTIONS = ["Nearest neighbour", "Bilinear", "Biquadratic", "Bicubic", "Biquartic", "Biquintic"]
    EDGE_FILL_OPTIONS = ["Zero", "Repeat", "Reflect"]
    edge_fill_options_scipy_argument = ['constant', 'edge', 'reflect']

    def __init__(self):
        super().__init__(Node.TYPE_REGISTER)  # Was: super(LoadDataNode, self).__init__()
        self.size = [230, 185]

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.image_in = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.INPUT, parent = self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)
        self.connectable_attributes.append(self.image_in)

        # Set up node-specific vars
        self.buffer_last_output = True
        self.register_method = 0
        self.reference_method = 1
        self.reference_image = None
        self.frame = 0
        self.roi = [0, 0, 0, 0]

        # StackReg vars
        self.sr = StackReg(StackReg.TRANSLATION)

        # CV vars
        self.orb = None
        self.orb_n_actual = 0
        self.orb_n_requested = 500
        self.orb_keep = 0.7
        self.orb_confidence = 0.99
        self.orb_method = 0

        # Advanced
        self.interpolation = 1
        self.edge_fill = 0
        self.preserve_range = False

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _c, self.use_roi = imgui.checkbox("use ROI", self.use_roi)
            self.any_change = self.any_change or _c
            imgui.push_item_width(140)
            _method_changed, self.register_method = imgui.combo("Method", self.register_method, RegisterNode.METHODS)
            _reference_changed, self.reference_method = imgui.combo("Reference", self.reference_method, RegisterNode.REFERENCES)
            imgui.pop_item_width()

            _frame_changed = False
            if self.reference_method == 1:
                imgui.push_item_width(50)
                _frame_changed, self.frame = imgui.input_int("Template frame", self.frame, 0, 0)
                imgui.pop_item_width()
            if self.reference_method == 0:
                imgui.new_line()
                self.image_in.render_start()
                self.image_in.render_end()

            if self.register_method == 0:
                pass  # no options (yet) for TurboReg
            elif self.register_method == 1:
                imgui.push_item_width(110)
                _c, self.orb_method = imgui.combo("Estimator", self.orb_method, ["Random sample consensus", "Least median of squares"])
                self.any_change = self.any_change or _c
                imgui.pop_item_width()
                imgui.push_item_width(90)
                _c, self.orb_n_requested = imgui.input_int("# features", self.orb_n_requested, 100, 100)
                self.any_change = self.any_change or _c
                _c, self.orb_keep = imgui.input_float("% matches to keep", self.orb_keep, format = "%.2f")
                self.any_change = self.any_change or _c
                _c, self.orb_confidence = imgui.input_float("confidence", self.orb_confidence, format = "%.2f")
                self.any_change = self.any_change or _c
                if _c:
                    self.orb_confidence = min([1.0, max([0.1, self.orb_confidence])])
                imgui.pop_item_width()

            self.any_change = self.any_change or _method_changed or _reference_changed or _frame_changed
            if self.any_change:
                self.reference_image = None

            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()

    def render_advanced(self):
        imgui.push_item_width(110)
        _c, self.interpolation = imgui.combo("Interpolation", self.interpolation, RegisterNode.INTERPOLATION_OPTIONS)
        _c, self.edge_fill = imgui.combo("Edges", self.edge_fill, RegisterNode.EDGE_FILL_OPTIONS)
        imgui.pop_item_width()
        Node.tooltip("Set how to fill pixels falling outside of the original image. Edge: clamp the edge\n"
                     "values, Reflect: reflect the image along the boundaries of the original image.")
        _c, self.preserve_range = imgui.checkbox("Preserve range", self.preserve_range)
        Node.tooltip("When checked, the intensity range of the output, registered image is fixed as the\n"
                     "same range as that of the original image.")

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            input_img = data_source.get_image(idx)
            if self.reference_image is None:
                # Get reference frame according to specified pairing method
                if self.reference_method == 2:
                    self.reference_image = data_source.get_image(idx - 1).load()
                elif self.reference_method == 0:
                    self.reference_image = self.image_in.get_incoming_node().get_image(idx=None).load()
                elif self.reference_method == 1:
                    self.reference_image = data_source.get_image(self.frame).load()

            # Perform registration according to specified registration method

            if self.reference_image is not None:
                template = self.reference_image
                image = input_img.load()
                if self.use_roi:
                    template = template[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                    image = image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                if self.register_method == 0:
                    tmat = self.sr.register(template, image)
                    input_img.translation = [tmat[0][2], tmat[1][2]]
                elif self.register_method == 1:
                    ## make new ORB if necessary
                    if self.orb_n_actual != self.orb_n_requested:
                        self.orb = cv2.ORB_create(nfeatures=self.orb_n_requested)
                    tmat = self._orb_register(template, image, keep=self.orb_keep, confidence=self.orb_confidence,
                                              method=self.orb_method)
                    input_img.translation = tmat
            else:
                NodeEditor.set_error(Exception(), "RegisterNode: reference image was None.")

            if self.reference_method == 2:
                self.reference_image = None
            input_img.bake_transform(interpolation=self.interpolation, edges=RegisterNode.edge_fill_options_scipy_argument[self.edge_fill], preserve_range=self.preserve_range)
            return input_img

    def pre_save_impl(self):
        self.reference_image = None

    def _orb_register(self, ref, img, keep=0.7, confidence=0.99, method=0):
        """

        :param ref: reference image
        :param img: image to be registered
        :param keep: the fraction (0.0 to 1.0) of matches to use to estimate the transformation - e.g. 0.1: keep top 10% matches
        :param confidence: confidence parameter of cv2.estimateAffinePartial2d
        :param method: 0 for RANSAC, 1 for LMEDS
        :return: list with two elements: x shift and y shift, in pixels.
        """
        cmin = np.amin(img)
        cmax = np.amax(img)
        ref8 = 255 * (ref - cmin) / (cmax - cmin)
        ref8 = ref8.astype(np.uint8)
        img8 = 255 * (img - cmin) / (cmax - cmin)
        img8 = img8.astype(np.uint8)

        kp1, d1 = self.orb.detectAndCompute(ref8, None)
        kp2, d2 = self.orb.detectAndCompute(img8, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = list(matcher.match(d1, d2))
        matches.sort(key=lambda x: x.distance)
        matches = matches[:int(len(matches) * keep)]
        n_matches = len(matches)

        p1 = np.zeros((n_matches, 2))
        p2 = np.zeros((n_matches, 2))

        for i in range(len(matches)):
            p1[i, :] = kp1[matches[i].queryIdx].pt
            p2[i, :] = kp2[matches[i].trainIdx].pt

        tmat, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC if method == 0 else cv2.LMEDS,
                                                   confidence=confidence)
        print(tmat)
        return [tmat[0][2], tmat[1][2]]


class GetImageNode(Node):
    IMAGE_MODES = ["By frame", "Time projection"]
    PROJECTIONS = ["Average", "Minimum", "Maximum", "St. dev."]

    def __init__(self):
        super().__init__(Node.TYPE_GET_IMAGE)
        self.size = [200, 120]
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT,parent=self)
        self.image_out = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.OUTPUT, parent=self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.image_out)

        self.mode = 0
        self.projection = 0
        self.frame = 0
        self.image = None
        self.load_data_source = None
        self.pixel_size = 1

        self.roi = [0, 0, 0, 0]

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.image_out.render_start()
            self.dataset_in.render_end()
            self.image_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.push_item_width(140)
            _c, self.mode = imgui.combo("Mode", self.mode, GetImageNode.IMAGE_MODES)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(80)
            if self.mode == 0:
                _c, self.frame = imgui.input_int("Frame nr.", self.frame, 0, 0)
                self.any_change = self.any_change or _c
            elif self.mode == 1:
                _c, self.projection = imgui.combo("Projection", self.projection, GetImageNode.PROJECTIONS)
                if _c:
                    self.image = None
                self.any_change = self.any_change or _c
            imgui.pop_item_width()
            if self.any_change:
                self.configure_settings()

            super().render_end()

    def configure_settings(self):
        datasource = self.dataset_in.get_incoming_node()
        if datasource:
            try:
                if NodeEditor.profiling:
                    self.profiler_count += 1
                if self.mode == 0:
                    image_in = datasource.get_image(self.frame)
                    self.image = image_in.load()
                    self.pixel_size = image_in.pixel_size
                elif self.mode == 1:
                    load_data_node = Node.get_source_load_data_node(self)
                    self.pixel_size = load_data_node.dataset.pixel_size
                    load_data_node.load_on_the_fly = False
                    self.load_data_source = load_data_node
            except Exception as e:
                NodeEditor.set_error(e, "GetImageNode error upon attempting to gen img.\n"+str(e))
        else:
            NodeEditor.set_error(Exception(), "GetImageNode missing input dataset.")
        self.any_change = True

    def on_update(self):
        if self.mode == 1 and self.image is None:
            if self.load_data_source is not None:
                if self.load_data_source.done_loading:
                    self.generate_projection()

    def generate_projection(self):
        data_source = self.dataset_in.get_incoming_node()
        frame = data_source.get_image(0)
        n_frames = Node.get_source_load_data_node(self).dataset.n_frames
        projection_image = np.zeros((frame.width, frame.height, n_frames))
        for i in range(n_frames):
            projection_image[:, :, i] = data_source.get_image(i).load()
        if self.projection == 0:
            self.image = np.average(projection_image, axis = 2)
        elif self.projection == 1:
            self.image = np.min(projection_image, axis = 2)
        elif self.projection == 2:
            self.image = np.max(projection_image, axis = 2)
        elif self.projection == 3:
            self.image = np.std(projection_image, axis = 2)
        self.any_change = True

    def get_image_impl(self, idx=None):
        if NodeEditor.profiling:
            self.profiler_count -= 1
        if self.any_change:
            self.configure_settings()
        if self.image is not None:
            out_frame = Frame("virtual_frame")
            out_frame.data = self.image
            out_frame.pixel_size = self.pixel_size
            return out_frame


class ImageCalculatorNode(Node):
    ## Note: the output dataset has all the metadata of dataset_in_a
    OPERATIONS = ["Add", "Subtract", "Divide", "Multiply"]

    def __init__(self):
        super().__init__(Node.TYPE_IMAGE_CALCULATOR)
        self.size = [230, 105]
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, parent=self, allowed_partner_types=[ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.TYPE_IMAGE])
        self.input_b = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, parent=self, allowed_partner_types=[ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.TYPE_IMAGE])
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)
        self.image_out = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.OUTPUT, parent=self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.input_b)
        self.connectable_attributes.append(self.dataset_out)
        self.connectable_attributes.append(self.image_out)

        self.operation = 1

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            if self.dataset_in.current_type == ConnectableAttribute.TYPE_IMAGE:
                self.image_out.render_start()
                self.image_out.render_end()
            else:
                self.dataset_out.render_start()
                self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            self.input_b.render_start()
            self.input_b.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.push_item_width(90)
            _c, self.operation = imgui.combo("Operation", self.operation, ImageCalculatorNode.OPERATIONS)
            self.any_change = self.any_change | _c
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        try:
            source_a = self.dataset_in.get_incoming_node()
            source_b = self.input_b.get_incoming_node()
            if source_a and source_b:
                img_a = source_a.get_image(idx)
                img_b = source_b.get_image(idx)
                img_a_pxd = img_a.load()
                img_b_pxd = img_b.load()
                img_out = None
                if self.operation == 0:
                    img_out = img_a_pxd + img_b_pxd
                elif self.operation == 1:
                    img_out = img_a_pxd - img_b_pxd
                elif self.operation == 2:
                    img_out = img_a_pxd / img_b_pxd
                elif self.operation == 3:
                    img_out = img_a_pxd * img_b_pxd
                if self.dataset_in.current_type != ConnectableAttribute.TYPE_IMAGE:
                    img_a.data = img_out
                    return img_a
                else:
                    virtual_frame = Frame("virtual_frame")
                    virtual_frame.data = img_out.astype(np.uint16)
                    return virtual_frame
        except Exception as e:
            NodeEditor.set_error(Exception(), "ImageCalculatorNode error:\n"+str(e))


class SpatialFilterNode(Node):
    FILTERS = ["Wavelet", "Gaussian", "Median"]
    WAVELETS = dict()
    WAVELETS["Haar"] = 'haar'
    WAVELETS["Symlet 2"] = 'sym2'
    WAVELETS["Symlet 3"] = 'sym3'
    WAVELETS["Daubechies 2"] = 'db2'
    WAVELETS["Biorthogonal 1.3"] = 'bior1.3'
    WAVELETS["Reverse biorthogonal 2.2"] = 'rbio2.2'
    WAVELETS["Other..."] = None


    WAVELET_NAMES = list(WAVELETS.keys())
    WAVELET_OTHER_IDX = WAVELET_NAMES.index("Other...")
    def __init__(self):
        super().__init__(Node.TYPE_SPATIAL_FILTER)  # Was: super(LoadDataNode, self).__init__()
        self.size = [210, 130]

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)
        # Set up node-specific vars
        self.filter = 1
        self.level = 1
        self.sigma = 2.0
        self.kernel = 3
        self.wavelet = 0
        self.custom_wavelet = "bior6.8"

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.push_item_width(140)
            _c, self.filter = imgui.combo("Filter", self.filter, SpatialFilterNode.FILTERS)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(70)
            if self.filter == 0:
                imgui.push_item_width(140)
                _c, self.wavelet = imgui.combo("Wavelet", self.wavelet, SpatialFilterNode.WAVELET_NAMES)
                imgui.pop_item_width()
                self.any_change = self.any_change or _c
                if self.wavelet == SpatialFilterNode.WAVELET_OTHER_IDX:
                    _c, self.custom_wavelet = imgui.input_text("pywt name", self.custom_wavelet, 16)
                    self.any_change = self.any_change or _c
                _c, self.level = imgui.input_int("Level", self.level, 0, 0)
                self.any_change = self.any_change or _c
            elif self.filter == 1:
                _c, self.sigma = imgui.input_float("Sigma (px)", self.sigma, 0.0, 0.0, format="%.1f")
                self.any_change = self.any_change or _c
            elif self.filter == 2:
                _c, self.kernel = imgui.input_int("Kernel (px)", self.kernel, 0, 0)
                if self.kernel % 2 == 0:
                    self.kernel += 1
                self.any_change = self.any_change or _c
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            input_image = data_source.get_image(idx)
            if self.filter == 0:
                chosen_wavelet = SpatialFilterNode.WAVELETS[SpatialFilterNode.WAVELET_NAMES[self.wavelet]]
                if chosen_wavelet is None:
                    chosen_wavelet = self.custom_wavelet
                in_pxd = input_image.load()
                input_image.data= pywt.swt2(in_pxd, wavelet=chosen_wavelet, level=self.level, norm=True, trim_approx=True)[0]
            elif self.filter == 1:
                input_image.data = gaussian_filter(input_image.load(), self.sigma)
            elif self.filter == 2:
                input_image.data = medfilt(input_image.load(), self.kernel)
            return input_image
        else:
            return None


class TemporalFilterNode(Node):
    FILTERS = ["Forward difference", "Backward difference", "Central difference", "Grouped difference", "Windowed average"]
    NEGATIVE_MODES = ["Absolute", "Zero", "Retain"]
    INCOMPLETE_GROUP_MODES = ["Discard"]

    def __init__(self):
        super().__init__(Node.TYPE_TEMPORAL_FILTER)  # Was: super(LoadDataNode, self).__init__()
        self.size = [250, 220]

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)

        self.filter = 0
        self.negative_handling = 1
        self.incomplete_group_handling = 0
        self.skip = 1
        self.group_size = 11
        self.group_background_index = 1
        self.window = 3

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(170)
            _c, self.filter = imgui.combo("Filter", self.filter, TemporalFilterNode.FILTERS)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(110)
            _c, self.negative_handling = imgui.combo("Negative handling", self.negative_handling, TemporalFilterNode.NEGATIVE_MODES)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(90)
            if self.filter == 0 or self.filter == 1 or self.filter == 2:
                _c, self.skip = imgui.input_int("Step (# frames)", self.skip, 0, 0)
                self.any_change = self.any_change or _c
            elif self.filter == 3:
                _c, self.group_size = imgui.input_int("Images per cycle", self.group_size, 0, 0)
                self.any_change = self.any_change or _c
                _c, self.group_background_index = imgui.input_int("Background index", self.group_background_index, 0, 0)
                self.any_change = self.any_change or _c
                _c, self.incomplete_group_handling = imgui.combo("Incomplete groups", self.incomplete_group_handling, TemporalFilterNode.INCOMPLETE_GROUP_MODES)
                self.any_change = self.any_change or _c
            elif self.filter == 4:
                _c, self.window = imgui.input_int("Window size", self.window, 0, 0)
                self.any_change = self.any_change or _c
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            pxd = None
            if self.filter == 0:
                pxd = data_source.get_image(idx + self.skip).load() - data_source.get_image(idx).load()
            elif self.filter == 1:
                pxd = data_source.get_image(idx).load() - data_source.get_image(idx - self.skip).load()
            elif self.filter == 2:
                pxd = data_source.get_image(idx + self.skip).load() - data_source.get_image(idx - self.skip).load()
            elif self.filter == 3:
                pxd = data_source.get_image(idx // self.group_size).load() - data_source.get_image(self.group_size * (idx // self.group_size) + self.group_background_index).load()
            elif self.filter == 4:
                pxd = np.zeros_like(data_source.get_image(idx).load())
                for i in range(-self.window, self.window + 1):
                    pxd += data_source.get_image(idx + i).load()
                pxd /= (2 * self.window + 1)

            if self.negative_handling == 0:
                pxd = np.abs(pxd)
            elif self.negative_handling == 1:
                pxd[pxd < 0] = 0
            elif self.negative_handling == 2:
                pass

            out_image = data_source.get_image(idx)
            out_image.data = pxd
            return out_image


class FrameShiftNode(Node):
    def __init__(self):
        super().__init__(Node.TYPE_FRAME_SHIFT)
        self.size = [140, 100]

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)

        self.shift = 0

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(80)
            _c, self.shift = imgui.input_int("shift", self.shift, 1, 10)
            self.any_change = _c or self.any_change

            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            return data_source.get_image(idx + self.shift)


class FrameSelectionNode(Node):
    def __init__(self):
        super().__init__(Node.TYPE_FRAME_SELECTION)  # Was: super(LoadDataNode, self).__init__()
        self.size = [200, 200]

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            super().render_end()


class ReconstructionRendererNode(Node):
    COLOUR_MODE = ["RGB, LUT"]

    def __init__(self):
        super().__init__(Node.TYPE_RECONSTRUCTOR)  # Was: super(LoadDataNode, self).__init__()
        self.size = [250, 200]

        self.reconstruction_in = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.INPUT, parent = self)
        self.image_out = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.OUTPUT, parent = self)

        self.connectable_attributes.append(self.reconstruction_in)
        self.connectable_attributes.append(self.image_out)

        self.magnification = 10
        self.default_sigma = 30.0
        self.fix_sigma = False

        self.reconstructor = Reconstructor()
        self.latest_image = None

        self.original_pixel_size = 100.0
        self.reconstruction_pixel_size = 10.0
        self.reconstruction_image_size = [1, 1]
        self.paint_particles = False
        self.paint_currently_applied = False

        paint_in = ConnectableAttribute(ConnectableAttribute.TYPE_COLOUR, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes.append(paint_in)
        self.particle_painters = [paint_in]
        self.does_profiling_time = False
        self.does_profiling_count = False

    def render(self):
        if super().render_start():
            self.reconstruction_in.render_start()
            self.image_out.render_start()
            self.reconstruction_in.render_end()
            self.image_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(100)
            _mag_changed, self.magnification = imgui.input_int("Magnification", self.magnification, 1, 1)
            self.magnification = max([self.magnification, 1])
            imgui.text(f"Final pixel size: {self.original_pixel_size / self.magnification:.1f}")
            imgui.text(f"Final image size: {self.reconstruction_image_size[0]} x {self.reconstruction_image_size[1]} px")

            _c, self.fix_sigma = imgui.checkbox("Force uncertainty", self.fix_sigma)


            if self.fix_sigma:
                imgui.same_line(spacing=10)
                imgui.push_item_width(50)
                _, self.default_sigma = imgui.input_float(" nm", self.default_sigma, 0, 0, format="%.1f")
                imgui.pop_item_width()

            # Colourize options
            _c, self.paint_particles = imgui.checkbox("Paint particles", self.paint_particles)
            if _c and not self.paint_particles:
                for i in range(len(self.particle_painters) - 1):
                    self.particle_painters[i].delete()

            imgui.spacing()
            if self.paint_particles:
                for connector in self.particle_painters:
                    # add / remove slots
                    if connector.check_connect_event():
                        new_slot = ConnectableAttribute(ConnectableAttribute.TYPE_COLOUR, ConnectableAttribute.INPUT, parent=self)
                        self.particle_painters.append(new_slot)
                        self.connectable_attributes.append(new_slot)
                    elif connector.check_disconnect_event():
                        self.particle_painters.remove(connector)
                        self.connectable_attributes.remove(connector)

                    # render blobs
                    imgui.new_line()
                    connector.render_start()
                    connector.render_end()

            _cw = imgui.get_content_region_available_width()
            imgui.new_line()
            imgui.same_line(spacing = _cw / 2 - 50 / 2)
            if imgui.button("Render", 50, 30):
                self.build_reconstruction()


            if _mag_changed:
                self.original_pixel_size = Node.get_source_load_data_node(self).dataset.pixel_size
                roi = self.get_particle_data().reconstruction_roi
                img_width = int((roi[2] - roi[0]) * self.magnification)
                img_height = int((roi[3] - roi[1]) * self.magnification)
                self.reconstruction_image_size = (img_width, img_height)

            super().render_end()

    def update_pixel_size(self):
        image_width = int((self.get_particle_data().reconstruction_roi[2] - self.get_particle_data().reconstruction_roi[0]) * self.magnification)
        image_height = int((self.get_particle_data().reconstruction_roi[3] - self.get_particle_data().reconstruction_roi[1]) * self.magnification)
        self.reconstruction_image_size = (image_width, image_height)
        new_pixel_size = self.original_pixel_size / self.magnification
        if new_pixel_size != self.reconstruction_pixel_size:
            self.reconstruction_pixel_size = new_pixel_size
            self.reconstructor.set_pixel_size(self.reconstruction_pixel_size)

    def build_reconstruction(self):
        try:
            self.original_pixel_size = Node.get_source_load_data_node(self).dataset.pixel_size
            self.update_pixel_size()
            self.reconstructor.set_pixel_size(self.original_pixel_size / self.magnification)
            self.reconstructor.set_image_size(self.reconstruction_image_size)
            datasource = self.reconstruction_in.get_incoming_node()
            if datasource:
                particle_data = datasource.get_particle_data()
                self.reconstructor.set_particle_data(particle_data)
                self.reconstructor.set_camera_origin([-particle_data.reconstruction_roi[0] * self.magnification, -particle_data.reconstruction_roi[1] * self.magnification])

                ## Apply colours
                if self.paint_particles:
                    if len(self.particle_painters) > 0:
                        for particle in particle_data.particles:
                            particle.colour = np.asarray([0.0, 0.0, 0.0])
                    for i in range(0, len(self.particle_painters) - 1):
                        self.particle_painters[i].get_incoming_node().apply_paint_to_particledata(particle_data)
                    self.paint_currently_applied = True
                else:
                    if self.paint_currently_applied:
                        for particle in particle_data.particles:
                            particle.colour = np.asarray([1.0, 1.0, 1.0])
                    self.paint_currently_applied = False

                if self.reconstructor.particle_data.empty:
                    return None
                else:
                    self.latest_image = self.reconstructor.render(fixed_uncertainty=(self.default_sigma if self.fix_sigma else None))
                    self.any_change = True
            else:
                self.latest_image = None
        except Exception as e:
            NodeEditor.set_error(e, "Error building reconstruction.\n"+str(e))

    def get_image_impl(self, idx=None):
        if self.latest_image is not None:
            img_wrapper = Frame("super-resolution reconstruction virtual frame")
            img_wrapper.data = self.latest_image
            img_wrapper.pixel_size = self.reconstruction_pixel_size
            return img_wrapper
        else:
            return None

    def get_particle_data_impl(self):
        datasource = self.reconstruction_in.get_incoming_node()
        if datasource:
            return datasource.get_particle_data()
        else:
            return ParticleData()

    def on_gain_focus(self):
        self.original_pixel_size = Node.get_source_load_data_node(self).dataset.pixel_size
        roi = self.get_particle_data().reconstruction_roi
        img_width = int((roi[2] - roi[0]) * self.magnification)
        img_height = int((roi[3] - roi[1]) * self.magnification)
        self.reconstruction_image_size = (img_width, img_height)

    def pre_save_impl(self):
        NodeEditor.pickle_temp["latest_image"] = self.latest_image
        self.latest_image = None

    def post_save_impl(self):
        self.latest_image = NodeEditor.pickle_temp["latest_image"]


class ParticlePainterNode(Node):

    def __init__(self):
        super().__init__(Node.TYPE_PARTICLE_PAINTER)  # Was: super(LoadDataNode, self).__init__()
        self.size = [270, 240]

        self.reconstruction_in = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.INPUT, parent = self)
        self.colour_out = ConnectableAttribute(ConnectableAttribute.TYPE_COLOUR, ConnectableAttribute.OUTPUT, parent = self)

        self.connectable_attributes.append(self.reconstruction_in)
        self.connectable_attributes.append(self.colour_out)

        self.parameter = 0
        self.available_parameters = ["Fixed colour"]
        self.colour_min = (0.0, 0.0, 0.0)
        self.colour_max = (1.0, 1.0, 1.0)
        self.histogram_values = np.asarray([0, 0]).astype('float32')
        self.histogram_bins = np.asarray([0, 0]).astype('float32')
        self.histogram_initiated = False
        self.min = 0
        self.max = 0
        self.paint_dry = True  # flag to notify whether settings were changed that would result in different particle colours.
        self.returns_image = False
        self.does_profiling_count = False

    def render(self):
        if super().render_start():
            self.reconstruction_in.render_start()
            self.colour_out.render_start()
            self.reconstruction_in.render_end()
            self.colour_out.render_end()

            if self.reconstruction_in.check_connect_event() or self.reconstruction_in.check_disconnect_event():
                self.init_histogram_values()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(160)
            _c, self.parameter = imgui.combo("Parameter", self.parameter, self.available_parameters)
            self.paint_dry = self.paint_dry or _c
            if _c:
                self.get_histogram_values()
                self.min = self.histogram_bins[0]
                self.max = self.histogram_bins[-1]
            imgui.pop_item_width()

            if self.parameter == 0:
                imgui.same_line()
                _c, self.colour_min = imgui.color_edit3("##Colour_min", *self.colour_min, imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL)
                self.colour_max = self.colour_max
                self.paint_dry = self.paint_dry or _c
            else:
                _c, self.colour_min = imgui.color_edit3("##Colour_min", *self.colour_min, imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL)
                self.paint_dry = self.paint_dry or _c
                imgui.same_line(spacing = 20)
                imgui.text("min")
                imgui.same_line(spacing=20)
                imgui.text("max")
                imgui.same_line(spacing = 20)
                _c, self.colour_max = imgui.color_edit3("##Colour_max", *self.colour_max, imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_SIDE_PREVIEW)
                self.paint_dry = self.paint_dry or _c

                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *self.colour_max)
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, *self.colour_max)
                content_width = imgui.get_content_region_available_width()
                imgui.plot_histogram("##hist", self.histogram_values, graph_size = (content_width, 40))
                imgui.text("{:.2f}".format(self.histogram_bins[0]))
                imgui.same_line(position = content_width - imgui.get_font_size() * len(str(self.histogram_bins[-1])) / 2)
                imgui.text("{:.2f}".format(self.histogram_bins[-1]))
                imgui.push_item_width(content_width)
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *self.colour_min)
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *self.colour_min)
                _c, self.min = imgui.slider_float("##min", self.min, self.histogram_bins[0], self.histogram_bins[-1], format = "min: %1.2f")
                self.paint_dry = self.paint_dry or _c
                imgui.pop_style_color(2)
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *self.colour_max)
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *self.colour_max)
                _c, self.max = imgui.slider_float("##max", self.max, self.histogram_bins[0], self.histogram_bins[-1], format="max: %1.2f")
                self.paint_dry = self.paint_dry or _c
                imgui.pop_style_color(4)
                imgui.pop_item_width()

            super().render_end()

    def get_histogram_values(self):
        datasource = self.reconstruction_in.get_incoming_node()
        if datasource:
            particledata = datasource.get_particle_data()
            if self.available_parameters[self.parameter] in list(particledata.histogram_counts.keys()):
                self.histogram_values = particledata.histogram_counts[self.available_parameters[self.parameter]]
                self.histogram_bins = particledata.histogram_bins[self.available_parameters[self.parameter]]

    def init_histogram_values(self):
        datasource = self.reconstruction_in.get_incoming_node()

        if datasource:
            self.histogram_initiated = True
            particledata = datasource.get_particle_data()
            if particledata.baked:
                self.available_parameters = ["Fixed colour"] + list(particledata.histogram_counts.keys())
                self.parameter = min([1, len(self.available_parameters)])
                self.histogram_values = particledata.histogram_counts[self.available_parameters[self.parameter]]
                self.histogram_bins = particledata.histogram_bins[self.available_parameters[self.parameter]]
                self.min = self.histogram_bins[0]
                self.max = self.histogram_bins[-1]

    def apply_paint_to_particledata(self, particledata):
        if NodeEditor.profiling:
            time_start = time.time()
        if self.parameter == 0:
            _colour = np.asarray(self.colour_max)
            for particle in particledata.particles:
                particle.colour += _colour
        else:
            i = 0
            c_min = np.asarray(self.colour_min)
            c_max = np.asarray(self.colour_max)
            values = particledata.parameter[self.available_parameters[self.parameter]]
            for particle in particledata.particles:
                fac = min([max([(values[i] - self.min) / self.max, 0.0]), 1.0])
                _colour = (1.0 - fac) * c_min + fac * c_max
                particle.colour += _colour
                i += 1
        if NodeEditor.profiling:
            self.profiler_time += time.time() - time_start

    def on_gain_focus(self):
        if self.histogram_initiated:
            self.get_histogram_values()


class ParticleDetectionNode(Node):
    METHODS = ["Local maximum"]
    THRESHOLD_OPTIONS = ["Value", "St. Dev.", "Mean", "Max", "Min"]

    def __init__(self):
        super().__init__(Node.TYPE_PARTICLE_DETECTION)
        self.size = [290, 205]

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.localizations_out = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.OUTPUT, parent=self)

        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.localizations_out)

        self.method = 0
        self.thresholding = 1
        self.threshold = 100
        self.sigmas = 2.0
        self.means = 3.0
        self.n_max = 2500
        self.d_min = 1
        self.max_fac = 0.75
        self.min_fac = 5.0
        self.roi = [0, 0, 0, 0]

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.localizations_out.render_start()
            self.dataset_in.render_end()
            self.localizations_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _c, self.use_roi = imgui.checkbox("use ROI", self.use_roi)
            self.any_change = self.any_change or _c
            imgui.push_item_width(160)
            _c, self.method = imgui.combo("Detection method", self.method, ParticleDetectionNode.METHODS)
            self.any_change = self.any_change or _c
            _c, self.thresholding = imgui.combo("Threshold method", self.thresholding, ParticleDetectionNode.THRESHOLD_OPTIONS)
            if self.thresholding == 0:
                _c, self.threshold = imgui.input_int("Value", self.threshold, 0, 0)
                self.any_change = self.any_change or _c
            elif self.thresholding == 1:
                _c, self.sigmas = imgui.slider_float("x Sigma", self.sigmas, 1.0, 5.0, format = "%.1f")
                self.any_change = self.any_change or _c
            elif self.thresholding == 2:
                _c, self.means = imgui.slider_float("x Mean", self.means, 0.1, 10.0, format = "%.1f")
                self.any_change = self.any_change or _c
            elif self.thresholding == 3:
                _c, self.max_fac = imgui.slider_float("x Max", self.max_fac, 0.0, 1.0, format="%.1f")
                self.any_change = self.any_change or _c
            elif self.thresholding == 4:
                _c, self.min_fac = imgui.slider_float("x Min", self.min_fac, 1.0, 10.0, format="%.1f")
                self.any_change = self.any_change or _c
            Node.tooltip("The final threshold value is determined by this factor x the metric (sigma, mean, etc.)\n"
                         "chosen above. In case of Threshold method 'Value', the value entered here is the final \n"
                         "threshold value. Note that the metric is calculated for the ROI only.")
            _c, self.n_max = imgui.slider_int("Max particles", self.n_max, 100, 2500)
            Node.tooltip("Maximum amount of particles output per frame. Ctrl + click to override the slider\n"
                         "and enter a custom value.")
            self.any_change = self.any_change or _c

            imgui.pop_item_width()
            imgui.push_item_width(100)
            _c, self.d_min = imgui.input_int("Minimum distance (px)", self.d_min, 1, 10)
            self.any_change = self.any_change or _c
            self.d_min = max([1, self.d_min])
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        source = self.dataset_in.get_incoming_node()
        if NodeEditor.profiling and self.frame_requested_by_image_viewer:
            self.profiler_count += 1
        if source is not None:
            # Find threshold value
            image_obj = source.get_image(idx)
            image = image_obj.load()
            if self.use_roi:
                image = image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
            threshold = self.threshold
            if self.thresholding == 1:
                threshold = self.sigmas * np.std(image)
            elif self.thresholding == 2:
                threshold = self.means * np.mean(image)
            elif self.thresholding == 3:
                threshold = self.max_fac * np.amax(image)
            elif self.thresholding == 4:
                threshold = self.min_fac * np.amin(image)
            # Perform requested detection method
            coordinates = peak_local_max(image, threshold_abs = threshold, num_peaks = self.n_max, min_distance = self.d_min)
            if self.use_roi:
                coordinates += np.asarray([self.roi[1], self.roi[0]])
                Node.get_source_load_data_node(self).dataset.reconstruction_roi = self.roi
            else:
                Node.get_source_load_data_node(self).dataset.reconstruction_roi = [0, 0, image.shape[0], image.shape[1]]
            image_obj.maxima = coordinates

            return image_obj

    def get_coordinates(self, idx=None):
        try:
            if NodeEditor.profiling:
                time_start = time.time()
                self.profiler_count += 1
            retval = self.get_image_impl(idx).maxima
            if NodeEditor.profiling:
                self.profiler_time += time.time() - time_start
            return retval
        except Exception as e:
            NodeEditor.set_error(e, "Error returning coordinates "+str(e))


class ExportDataNode(Node):

    def __init__(self):
        super().__init__(Node.TYPE_EXPORT_DATA)
        self.size = [210, 220]

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, self, [ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_RECONSTRUCTION])
        self.dataset_in.title = "Any"
        self.connectable_attributes.append(self.dataset_in)

        self.path = "..."
        self.roi = [0, 0, 0, 0]
        self.include_discarded_frames = False
        self.saving = False
        self.export_type = 0  # 0 for dataset, 1 for image.
        self.frames_to_load = list()
        self.n_frames_to_save = 1
        self.n_frames_saved = 0

        self.parallel = True
        self.returns_image = False
        self.does_profiling_count = False

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.dataset_in.render_end()

            self.export_type = -1
            if self.dataset_in.get_incoming_attribute_type() == ConnectableAttribute.TYPE_DATASET:
                self.export_type = 0
            elif self.dataset_in.get_incoming_attribute_type() == ConnectableAttribute.TYPE_IMAGE:
                self.export_type = 1
            elif self.dataset_in.get_incoming_attribute_type() == ConnectableAttribute.TYPE_RECONSTRUCTION:
                self.export_type = 2
            else:
                self.dataset_in.title = "Any"
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if self.export_type in [0, 1]:
                _, self.use_roi = imgui.checkbox("use ROI", self.use_roi)
            imgui.text("Output path")
            imgui.push_item_width(150)
            _, self.path = imgui.input_text("##intxt", self.path, 256, imgui.INPUT_TEXT_ALWAYS_OVERWRITE)
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("...", 26, 19):
                filename = filedialog.asksaveasfilename()
                if filename is not None:
                    if '.' in filename[-5:]:
                        filename = filename[:filename.rfind(".")]
                    self.path = filename
            if self.export_type == 0:
                _c, self.parallel = imgui.checkbox("Parallel", self.parallel)
            content_width = imgui.get_window_width()
            save_button_width = 100
            save_button_height = 40

            if self.saving:
                imgui.spacing()
                imgui.text("Export progress")
                self.progress_bar(self.n_frames_saved / self.n_frames_to_save)

            imgui.spacing()
            imgui.spacing()
            imgui.spacing()
            imgui.new_line()
            imgui.same_line(position=(content_width - save_button_width) // 2)
            if not self.saving:
                if imgui.button("Save", save_button_width, save_button_height):
                    self.do_save()
            else:
                if imgui.button("Cancel", save_button_width, save_button_height):
                    self.saving = False
            super().render_end()

    def get_img_and_save(self, idx):
        img_pxd = self.get_image_impl(idx)
        if self.use_roi:
            img_pxd = img_pxd[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        Image.fromarray(img_pxd).save(self.path+"/0"+str(idx)+".tif")

    def do_save(self):
        if NodeEditor.profiling:
            time_start = time.time()
        if self.export_type == 0:  # Save stack
            self.saving = True
            #if self.include_discarded_frames:
                #n_active_frames = Node.get_source_load_data_node(self).dataset.n_frames
                #self.frames_to_load = list(range(0, n_active_frames))
            self.frames_to_load = list()
            for i in range(Node.get_source_load_data_node(self).dataset.n_frames):
                self.frames_to_load.append(i)
            self.n_frames_to_save = len(self.frames_to_load)
            self.n_frames_saved = 0
            if not os.path.isdir(self.path):
                os.mkdir(self.path)
        elif self.export_type == 1:  # Save image
            img = self.dataset_in.get_incoming_node().get_image(idx=None)
            img_pxd = img.load()
            if self.use_roi:
                img_pxd = img_pxd[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
            try:
                util.save_tiff(img_pxd, self.path+".tif", pixel_size_nm=img.pixel_size)
            except Exception as e:
                NodeEditor.set_error(e, "Error saving image: "+str(e))
        elif self.export_type == 2: # Save particle data
            try:
                self.dataset_in.get_incoming_node().get_particle_data().save_as_csv(self.path+".csv")
            except Exception as e:
                NodeEditor.set_error(e, "Error saving .csv\n"+str(e))
        if NodeEditor.profiling:
            self.profiler_time += time.time() - time_start

    def on_update(self):
        if self.saving:
            try:
                if self.parallel:
                    indices = list()
                    for i in range(min([NodeEditor.batch_size, len(self.frames_to_load)])):
                        self.n_frames_saved += 1
                        indices.append(self.frames_to_load[-1])
                        self.frames_to_load.pop()
                    Parallel(n_jobs=NodeEditor.batch_size)(delayed(self.get_img_and_save)(index) for index in indices)
                else:
                    self.n_frames_saved += 1
                    self.get_img_and_save(self.frames_to_load[-1])
                    self.frames_to_load.pop()

                if len(self.frames_to_load) == 0:
                    self.saving = False
            except Exception as e:
                self.saving = False
                NodeEditor.set_error(e, "Error saving stack: \n"+str(e))

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            incoming_img = data_source.get_image(idx)
            img_pxd = incoming_img.load()
            incoming_img.clean()
            return img_pxd
        img_source = self.image_in.get_incoming_node()
        if img_source:
            return img_source.get_image(idx)


class BinImageNode(Node):
    MODES = ["Average", "Median", "Min", "Max", "Sum"]

    def __init__(self):
        super().__init__(Node.TYPE_BIN_IMAGE)
        self.size = [170, 120]
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, self, [ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_DATASET])
        self.output = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.OUTPUT, self, [ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_DATASET])
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.output)

        self.factor = 2
        self.mode = 0

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.output.render_start()
            self.dataset_in.render_end()
            self.output.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(100)
            _c, self.mode = imgui.combo("Method", self.mode, BinImageNode.MODES)
            self.any_change = _c or self.any_change
            imgui.pop_item_width()
            imgui.push_item_width(60)
            _c, self.factor = imgui.input_int("Bin factor", self.factor, 0, 0)
            self.any_change = _c or self.any_change
            self.factor = max([self.factor, 1])
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            image_in = data_source.get_image(idx)
            pxd = image_in.load()
            width, height = pxd.shape
            pxd = pxd[:self.factor * (width // self.factor), :self.factor * (height // self.factor)]
            if self.mode == 0:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).mean(2).mean(0)
            elif self.mode == 1:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).median(2).median(0)
            elif self.mode == 2:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).min(2).min(0)
            elif self.mode == 3:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).max(2).max(0)
            elif self.mode == 4:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).sum(2).sum(0)
            image_in.data = pxd
            return image_in


class ParticleFittingNode(Node):
    RANGE_OPTIONS = ["All frames", "Current frame only", "Custom range"]
    ESTIMATORS = ["Least squares (GPU)", "Maximum likelihood (GPU)", "No estimator (CPU, no parallel)"]
    PSFS = ["Gaussian", "Elliptical Gaussian"]

    def __init__(self):
        super().__init__(Node.TYPE_PARTICLE_FITTING)

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.localizations_in = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.INPUT, parent=self)
        self.reconstruction_out = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.OUTPUT, parent=self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.reconstruction_out)
        self.connectable_attributes.append(self.localizations_in)

        self.size = [300, 270]
        self.range_option = 1
        self.range_min = 0
        self.range_max = 1
        self.estimator = 1
        self.crop_radius = 3
        self.initial_sigma = 1.6
        self.fitting = False
        self.n_to_fit = 1
        self.n_fitted = 0
        self.frames_to_fit = list()
        self.particle_data = ParticleData()

        self.time_start = 0
        self.time_stop = 0

        self.intensity_min = 100.0
        self.intensity_max = -1.0
        self.sigma_min = 1.0
        self.sigma_max = 10.0
        self.offset_min = 0.0
        self.offset_max = -1.0

        self.custom_bounds = False

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.reconstruction_out.render_start()
            self.dataset_in.render_end()
            self.reconstruction_out.render_end()
            imgui.new_line()
            self.localizations_in.render_start()
            self.localizations_in.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _c, self.estimator = imgui.combo("Estimator", self.estimator, ParticleFittingNode.ESTIMATORS)
            self.any_change = _c or self.any_change
            imgui.push_item_width(80)
            if self.estimator in [0, 1]:
                _c, self.initial_sigma = imgui.input_float("Initial sigma (px)", self.initial_sigma, 0, 0, "%.1f")
                self.any_change = _c or self.any_change
                _c, self.crop_radius = imgui.input_int("Fitting radius (px)", self.crop_radius, 0, 0)
                self.any_change = _c or self.any_change
            imgui.pop_item_width()
            imgui.spacing()
            _c, self.range_option = imgui.combo("Range", self.range_option,
                                                ParticleFittingNode.RANGE_OPTIONS)
            self.any_change = _c or self.any_change
            if self.range_option == 2:
                imgui.push_item_width(80)
                _c, (self.range_min, self.range_max) = imgui.input_int2('[start, stop) index', self.range_min,
                                                                        self.range_max)
                self.any_change = _c or self.any_change
                imgui.pop_item_width()
            imgui.spacing()
            if self.fitting:
                self.progress_bar(self.n_fitted / self.n_to_fit)
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()

            _play_btn_clicked, _state = self.play_button()
            if _play_btn_clicked and _state:
                self.init_fit()
            elif _play_btn_clicked and self.fitting:
                self.fitting = False

            ## TODO: add camera offset and ADU per photon (to LOAD DATASET?)
            if not self.particle_data.empty:
                imgui.text("Reconstruction info")
                imgui.text(f"particles: {len(self.particle_data.particles)}")
                if not self.fitting:
                    imgui.text(f"x range: {self.particle_data.x_min / 1000.0:.1f} to {self.particle_data.x_max / 1000.0:.1f} um")
                    imgui.text(f"y range: {self.particle_data.y_min / 1000.0:.1f} to {self.particle_data.y_max / 1000.0:.1f} um")
            imgui.spacing()
            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()

    def render_advanced(self):
        _c, self.custom_bounds = imgui.checkbox("Use custom parameter bounds", self.custom_bounds)
        Node.tooltip("Edit the bounds for particle parameters intensity, sigma, and offset.")
        if self.custom_bounds:
            imgui.push_item_width(45)
            _c, self.sigma_min = imgui.input_float("min##sigma", self.sigma_min, 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle sigma to a minimum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            _c, self.sigma_max = imgui.input_float("max##sigma", self.sigma_max, 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle sigma to a maximum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            imgui.text("sigma (px)")

            _c, self.intensity_min = imgui.input_float("min##int", self.intensity_min, 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle intensity to a minimum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            _c, self.intensity_max = imgui.input_float("max##int", self.intensity_max, 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle intensity to a maximum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            imgui.text("intensity (counts)")

            _c, self.offset_min = imgui.input_float("min##offset", self.offset_min, 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle offset to a minimum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            _c, self.offset_max = imgui.input_float("max##offset", self.offset_max, 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle offset to a maximum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            imgui.text("offset (counts)")

            imgui.pop_item_width()

    def init_fit(self):
        try:
            self.time_start = time.time()
            dataset_source = Node.get_source_load_data_node(self)
            self.particle_data = ParticleData(dataset_source.pixel_size)
            self.particle_data.set_reconstruction_roi(dataset_source.dataset.reconstruction_roi)
            self.fitting = True
            self.frames_to_fit = list()
            if self.range_option == 0:
                dataset = dataset_source.dataset
                n_frames = dataset.n_frames
                self.frames_to_fit = list(range(0, n_frames))
            elif self.range_option == 2:
                self.frames_to_fit = list(range(self.range_min, self.range_max))
            elif self.range_option == 1:
                dataset = dataset_source.dataset
                self.frames_to_fit = [dataset.current_frame]
            self.n_to_fit = len(self.frames_to_fit)
            self.n_fitted = 0
        except Exception as e:
            self.fitting = False
            NodeEditor.set_error(e, "Error in init_fit: "+str(e))

    def on_update(self):
        try:
            if self.fitting:
                if len(self.frames_to_fit) == 0:
                    self.fitting = False
                    self.play = False
                    self.particle_data.bake()
                else:
                    fitted_frame = self.get_image(self.frames_to_fit[-1])
                    self.n_fitted += 1
                    if fitted_frame is not None:
                        particles = fitted_frame.particles
                        self.frames_to_fit.pop()
                        if not fitted_frame.discard:
                            self.particle_data += particles
        except Exception as e:
            self.fitting = False
            self.play = False
            NodeEditor.set_error(e, "Error in ParticleFitNode.on_update(self): "+str(e))

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        coord_source = self.localizations_in.get_incoming_node()
        if data_source and coord_source:
            frame = data_source.get_image(idx)
            coordinates = coord_source.get_coordinates(idx)
            frame.maxima = coordinates
            if frame.discard:
                frame.particles = None
                return frame
            particles = list()
            if self.estimator in [0, 1]:
                particles = pfit.frame_to_particles(frame, self.initial_sigma, self.estimator, self.crop_radius, constraints = [self.intensity_min, self.intensity_max, -1, -1, -1, -1, self.sigma_min, self.sigma_max, self.offset_min, self.offset_max])
            elif self.estimator == 2:
                particles = list()
                pxd = frame.load()
                for i in range(len(frame.maxima)):
                    intensity = pxd[frame.maxima[i, 0], frame.maxima[i, 1]]
                    x = frame.maxima[i, 0]
                    y = frame.maxima[i, 1]
                    particles.append(Particle(idx, x, y, 1.0, intensity))
            frame.particles = particles
            new_maxima = list()
            for particle in frame.particles:
                new_maxima.append([particle.y, particle.x])
            frame.maxima = new_maxima
            return frame

    def get_particle_data_impl(self):
        self.particle_data.clean()
        return self.particle_data

    def pre_save_impl(self):
        NodeEditor.pickle_temp["particle_data"] = self.particle_data
        self.particle_data = ParticleData()

    def post_save_impl(self):
        self.particle_data = NodeEditor.pickle_temp["particle_data"]


class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__(Node.TYPE_PARTICLE_FILTER)

        self.reconstruction_in = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.INPUT, parent=self)
        self.reconstruction_out = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.OUTPUT, parent=self)

        self.connectable_attributes.append(self.reconstruction_in)
        self.connectable_attributes.append(self.reconstruction_out)

        self.size = [310, 250]
        self.available_parameters = ["No data available"]
        self.filters = list()

        self.returns_image = False
        self.does_profiling_count = False

    def render(self):
        if super().render_start():
            self.reconstruction_in.render_start()
            self.reconstruction_out.render_start()
            self.reconstruction_in.render_end()
            self.reconstruction_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if self.reconstruction_in.check_connect_event():
                self.get_histogram_parameters()

            i = 0
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, 0.2, 0.2, 0.2, 1.0)
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, 0.2, 0.2, 0.2, 1.0)
            for pf in self.filters:
                imgui.push_id(f"pfid{i}")
                if imgui.button("-", 20, 20):
                    self.filters.remove(pf)
                    imgui.pop_id()
                    continue
                imgui.same_line()
                imgui.set_next_item_width(144)
                _c, pf.parameter = imgui.combo("Criterion", pf.parameter, self.available_parameters)
                if _c:
                    pf.set_data(*self.get_histogram_vals(self.available_parameters[pf.parameter]), self.available_parameters[pf.parameter])
                imgui.same_line(spacing = 10)
                _c, pf.invert = imgui.checkbox("NOT", pf.invert)
                Node.tooltip("If NOT is selected, the filter logic is inverted. Default\n"
                             "behaviour is: particle retained if min < parameter < max.\n"
                             "With NOT on, a particle is retained if max < parameter OR\nmin > parameter")

                pf.render_start()
                imgui.pop_id()
                i += 1
            content_width = imgui.get_window_content_region_width()
            imgui.new_line()
            imgui.same_line(content_width / 2 - 50)
            if imgui.button("Add filter", width = 100, height = 20):
                new_filter = ParticleFilterNode.Filter(self)
                new_filter.set_data(*self.get_histogram_vals(self.available_parameters[0]), self.available_parameters[0])
                self.filters.append(new_filter)
            imgui.pop_style_color(2)
            super().render_end()

    def on_gain_focus(self):
        self.get_histogram_parameters()

    def get_histogram_vals(self, parameter):
        datasource = self.reconstruction_in.get_incoming_node()
        if datasource:
            particledata = datasource.get_particle_data()
            return particledata.histogram_counts[parameter], particledata.histogram_bins[parameter]
        else:
            return np.asarray([0, 0]).astype('float32'), np.asarray([0, 1]).astype('float32')

    def get_histogram_parameters(self):
        datasource = self.reconstruction_in.get_incoming_node()
        if datasource:
            particledata = datasource.get_particle_data()
            self.available_parameters = list(particledata.histogram_counts.keys())

    def get_particle_data_impl(self):
        if NodeEditor.profiling:
            time_start = time.time()
        datasource = self.reconstruction_in.get_incoming_node()
        if datasource:
            pdata = datasource.get_particle_data()
            for pf in self.filters:
                pf.apply(pdata)
            if NodeEditor.profiling:
                self.profiler_time += time.time() - time_start
            return pdata
        else:
            if NodeEditor.profiling:
                self.profiler_time += time.time() - time_start
            return ParticleData()

    class Filter:
        def __init__(self, parent):
            self.parameter_key = ""
            self.vals = [0, 0]
            self.bins = [0, 0]
            self.min = 0
            self.max = 1
            self.parameter = 0
            self.invert = False
            self.parent = parent

        def set_data(self, vals, bins, prm_key):
            self.vals = vals
            self.bins = bins
            self.min = bins[0]
            self.max = bins[-1]
            self.parameter_key = prm_key

        def render(self):
            content_width = imgui.get_content_region_available_width()
            imgui.plot_histogram("##histogram", self.vals, graph_size = (content_width, 40))
            imgui.text("{:.2f}".format(self.bins[0]))
            imgui.same_line(position=content_width - imgui.get_font_size() * len(str(self.bins[-1])) / 2)
            imgui.text("{:.2f}".format(self.bins[-1]))
            imgui.push_item_width(content_width)
            _c, self.min = imgui.slider_float("##min", self.min, self.bins[0], self.bins[-1], "min: %1.2f")
            _c, self.max = imgui.slider_float("##max", self.max, self.bins[0], self.bins[-1], "max: %1.2f")
            imgui.pop_item_width()
            imgui.separator()
            imgui.spacing()

        def apply(self, particle_data_object):
            if NodeEditor.profiling:
                time_start = time.time()
            particle_data_object.apply_filter(self.parameter_key, self.min, self.max, logic_not=self.invert)
            if NodeEditor.profiling:
                self.parent.profiler_time += time.time() - time_start


class BakeStackNode(Node):
    RANGE_OPTIONS = ["All frames", "Custom range"]

    def __init__(self):
        super().__init__(Node.TYPE_BAKE_STACK)# make it take dataset AND coordinates.
        self.size = [230, 100]

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, self)
        self.coordinates_in = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.INPUT, self)
        self.coordinates_out = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.OUTPUT, self)

        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)
        self.connectable_attributes.append(self.coordinates_in)
        self.connectable_attributes.append(self.coordinates_out)

        self.parallel = True
        self.baking = False
        self.n_to_bake = 1
        self.n_baked = 0
        self.range_option = 0
        self.custom_range_min = 0
        self.custom_range_max = 1
        self.has_dataset = False
        self.dataset = Dataset()
        self.frames_to_bake = list()
        self.temp_dir_do_not_edit = "_srnodes_temp"+str(self.id)
        self.joblib_dir_counter = 0
        self.baked_at = 0
        self.has_coordinates = False
        self.coordinates = list()
        self.bake_coordinates = False

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.dataset_in.render_end()
            if self.has_dataset:
                self.dataset_out.render_start()
                self.dataset_out.render_end()
            if self.bake_coordinates:
                imgui.spacing()
                self.coordinates_in.render_start()
                self.coordinates_in.render_end()
                if self.has_coordinates:
                    self.coordinates_out.render_start()
                    self.coordinates_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _c, self.bake_coordinates = imgui.checkbox("Also bake coordinates", self.bake_coordinates)
            if _c and not self.bake_coordinates:
                self.coordinates_in.disconnect_all()

            if self.range_option == 1:
                imgui.push_item_width(80)
                _c, (self.custom_range_min, self.custom_range_max) = imgui.input_int2('[start, top) index', self.custom_range_min, self.custom_range_max)
                imgui.pop_item_width()
            _c, self.parallel = imgui.checkbox("Parallel processing", self.parallel)
            imgui.set_next_item_width(100)
            _c, self.range_option = imgui.combo("Range to bake", self.range_option, BakeStackNode.RANGE_OPTIONS)
            clicked, self.baking = self.play_button()
            if clicked and self.baking:
                self.init_bake()

            if self.baking:
                imgui.text("Baking process:")
                self.progress_bar(self.n_baked / self.n_to_bake)
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()

            if self.has_dataset:
                imgui.text(f"Output data was baked at: {self.baked_at}")
            super().render_end()

    def get_image_and_save(self, idx=None):
        datasource = self.dataset_in.get_incoming_node()
        if datasource:
            pxd = datasource.get_image(idx).load()
            Image.fromarray(pxd).save(self.temp_dir_do_not_edit + "/0" + str(idx) + ".tif")
            if self.bake_coordinates:
                coordsource = self.coordinates_in.get_incoming_node()
                coordinates = coordsource.get_coordinates(idx)
                return coordinates
            return False
        return False

    def get_image_impl(self, idx=None):
        if self.has_dataset and idx in range(0, self.dataset.n_frames):
            retimg = copy.deepcopy(self.dataset.get_indexed_image(idx))
            retimg.clean()
            if self.has_coordinates:
                retimg.maxima = self.coordinates[idx]
            return retimg
        else:
            datasource = self.dataset_in.get_incoming_node()
            if datasource:
                return datasource.get_image(idx)

    def get_coordinates(self, idx=None):
        if self.has_coordinates and idx in range(0, self.dataset.n_frames):
            return self.coordinates[idx]

    def init_bake(self):
        self.dataset = None
        if os.path.isdir(self.temp_dir_do_not_edit):
            for f in glob.glob(self.temp_dir_do_not_edit+"/*"):
                os.remove(f)
        else:
            os.mkdir(self.temp_dir_do_not_edit)
        self.has_dataset = False
        self.has_coordinates = False
        self.coordinates = list()
        if self.range_option == 0:
            dataset_source = Node.get_source_load_data_node(self)
            self.frames_to_bake = list(range(0, dataset_source.dataset.n_frames))
        elif self.range_option == 1:
            self.frames_to_bake = list(range(self.custom_range_min, self.custom_range_max))
        self.n_baked = 0
        self.baked_at = datetime.datetime.now().strftime("%H:%M")
        self.n_to_bake = max([1, len(self.frames_to_bake)])

    def on_update(self):
        if self.baking:
            if NodeEditor.profiling:
                time_start = time.time()
            try:
                if self.parallel:
                    indices = list()
                    for i in range(min([NodeEditor.batch_size, len(self.frames_to_bake)])):
                        self.n_baked += 1
                        indices.append(self.frames_to_bake[-1])
                        self.frames_to_bake.pop()
                    coordinates = Parallel(n_jobs=NodeEditor.batch_size)(delayed(self.get_image_and_save)(index) for index in indices)
                else:
                    index = self.frames_to_bake[-1]
                    self.frames_to_bake.pop()
                    coordinates = self.get_image_and_save(index)
                if self.bake_coordinates:
                    self.coordinates += coordinates
                if NodeEditor.profiling:
                    self.profiler_time += time.time() - time_start
                    self.profiler_count += len(indices)

                if len(self.frames_to_bake) == 0:
                    # Load dataset from temp dir.
                    self.dataset = Dataset(self.temp_dir_do_not_edit + "/00.tif")
                    print("Done baking - now loading from disk")
                    for frame in self.dataset.frames:
                        frame.load()
                    print("Done loading")
                    self.any_change = True
                    self.baking = False
                    self.play = False
                    self.has_dataset = True
                    if self.bake_coordinates:
                        self.coordinates.reverse()
                        self.has_coordinates = True
            except Exception as e:
                self.baking = False
                self.play = False
                NodeEditor.set_error(e, "Error baking stack: \n"+str(e))


class CropImageNode(Node):
    def __init__(self):
        super().__init__(Node.TYPE_CROP_IMAGE)
        self.size = [140, 100]
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT,parent=self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)

        self.roi = [0, 0, 0, 0]
        self.use_roi = True

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            if self.frame_requested_by_image_viewer:
                return data_source.get_image(idx)
            else:
                out_frame = data_source.get_image(idx).clone()
                pxd = out_frame.load()[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                out_frame.data = pxd
                out_frame.width, out_frame.height = out_frame.data.shape
                return out_frame
