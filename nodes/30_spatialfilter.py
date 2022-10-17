from node import *

import pywt
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.signal import medfilt

def create():
    return SpatialFilterNode()


class SpatialFilterNode(Node):
    title = "Spatial filter"
    group = "Image processing"
    colour = (44 / 255, 217 / 255, 158 / 255, 1.0)

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
        super().__init__()  # Was: super(LoadDataNode, self).__init__()
        self.size = 210

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)

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
