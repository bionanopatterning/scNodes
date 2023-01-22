from scNodes.core.node import *

import pywt
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt

def create():
    return SpatialFilterNode()


class SpatialFilterNode(Node):
    title = "Spatial filter"
    group = "Image processing"
    colour = (44 / 255, 217 / 255, 158 / 255, 1.0)
    sortid = 101

    FILTERS = ["Wavelet", "Gaussian", "Median", "Difference of Gaussians", "Derivative of Gaussian"]
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
        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)

        # Set up node-specific vars
        self.params["filter"] = 1
        self.params["level"] = 1
        self.params["sigma"] = 2.0
        self.params["kernel"] = 3
        self.params["wavelet"] = 0
        self.params["custom_wavelet"] = "bior6.8"
        self.params["dog_s1"] = 1.0
        self.params["dog_s2"] = 5.0
        self.params["deriv_sigma"] = 2.0
        self.params["deriv_order"] = 1

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_out"].render_start()
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_out"].render_end()
            self.connectable_attributes["dataset_in"].render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.push_item_width(140)
            _c, self.params["filter"] = imgui.combo("Filter", self.params["filter"], SpatialFilterNode.FILTERS)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(70)
            if self.params["filter"] == 0:
                imgui.push_item_width(140)
                _c, self.params["wavelet"] = imgui.combo("Wavelet", self.params["wavelet"], SpatialFilterNode.WAVELET_NAMES)
                imgui.pop_item_width()
                self.any_change = self.any_change or _c
                if self.params["wavelet"] == SpatialFilterNode.WAVELET_OTHER_IDX:
                    _c, self.params["custom_wavelet"] = imgui.input_text("pywt name", self.params["custom_wavelet"], 16)
                    self.any_change = self.any_change or _c
                    Node.tooltip("Enter any of the pywavelets dicrete wavelet names.\n"
                                 "See: http://wavelets.pybytes.com/ ")
                _c, self.params["level"] = imgui.input_int("Level", self.params["level"], 0, 0)
                self.any_change = self.any_change or _c
            elif self.params["filter"] == 1:
                _c, self.params["sigma"] = imgui.input_float("Sigma (px)", self.params["sigma"], 0.0, 0.0, format="%.1f")
                self.any_change = self.any_change or _c
            elif self.params["filter"] == 2:
                _c, self.params["kernel"] = imgui.input_int("Kernel (px)", self.params["kernel"], 0, 0)
                if self.params["kernel"] % 2 == 0:
                    self.params["kernel"] += 1
                self.any_change = self.any_change or _c
            elif self.params["filter"] == 3:
                _c, self.params["dog_s1"] = imgui.input_float("Sigma 1 (px)", self.params["dog_s1"], 0.0, 0.0, format="%.1f")
                self.any_change = self.any_change or _c
                _c, self.params["dog_s2"] = imgui.input_float("Sigma 2 (px)", self.params["dog_s2"], 0.0, 0.0, format="%.1f")
                self.any_change = self.any_change or _c
            elif self.params["filter"] == 4:
                _c, self.params["deriv_sigma"] = imgui.input_float("Sigma 1 (px)", self.params["deriv_sigma"], 0.0, 0.0, format="%.1f")
                self.any_change = self.any_change or _c
                _c, self.params["deriv_order"] = imgui.input_int("Order", self.params["deriv_order"], 0, 0)
                self.any_change = self.any_change or _c
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.connectable_attributes["dataset_in"].get_incoming_node()
        if data_source:
            input_image = data_source.get_image(idx)
            outframe = input_image.clone()
            pxd = input_image.load()
            if self.params["filter"] == 0:
                chosen_wavelet = SpatialFilterNode.WAVELETS[SpatialFilterNode.WAVELET_NAMES[self.params["wavelet"]]]
                w, h = pxd.shape
                pxd = pxd[:(w//2)*2, :(h//2)*2]
                if chosen_wavelet is None:
                    chosen_wavelet = self.params["custom_wavelet"]
                pxd = pywt.swt2(pxd, wavelet=chosen_wavelet, level=self.params["level"], norm=True, trim_approx=True)[0]
            elif self.params["filter"] == 1:
                pxd = gaussian_filter(pxd, self.params["sigma"])
            elif self.params["filter"] == 2:
                pxd = medfilt(pxd, self.params["kernel"])
            elif self.params["filter"] == 3:
                pxd = gaussian_filter(pxd, self.params["dog_s1"]) - gaussian_filter(pxd, self.params["dog_s2"])
            elif self.params["filter"] == 4:
                pxd = gaussian_filter(pxd, self.params["deriv_sigma"], order=self.params["deriv_order"])
            outframe.data = pxd
            return outframe
        else:
            return None
