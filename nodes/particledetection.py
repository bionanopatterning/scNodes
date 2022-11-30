from node import *
from skimage.feature import peak_local_max

def create():
    return ParticleDetectionNode()


class ParticleDetectionNode(Node):
    title = "Particle detection"
    group = "Reconstruction"
    colour = (230 / 255, 174 / 255, 13 / 255, 1.0)
    sortid = 1000

    METHODS = ["Local maximum"]
    THRESHOLD_OPTIONS = ["Value", "St. Dev.", "Mean", "Max", "Min"]

    def __init__(self):
        super().__init__()
        self.size = 290

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.localizations_out = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.OUTPUT, parent=self)


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
            self.any_change = self.any_change or _c
            if self.thresholding == 0:
                _c, self.threshold = imgui.input_int("Value", self.threshold, 0, 0)
                self.any_change = self.any_change or _c
            elif self.thresholding == 1:
                _c, self.sigmas = imgui.slider_float("x Sigma", self.sigmas, 1.0, 5.0, format = "%.2f")
                self.any_change = self.any_change or _c
            elif self.thresholding == 2:
                _c, self.means = imgui.slider_float("x Mean", self.means, 0.1, 10.0, format = "%.2f")
                self.any_change = self.any_change or _c
            elif self.thresholding == 3:
                _c, self.max_fac = imgui.slider_float("x Max", self.max_fac, 0.0, 1.0, format="%.2f")
                self.any_change = self.any_change or _c
            elif self.thresholding == 4:
                _c, self.min_fac = imgui.slider_float("x Min", self.min_fac, 1.0, 10.0, format="%.2f")
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
        if cfg.profiling and self.FRAME_REQUESTED_BY_IMAGE_VIEWER:
            self.profiler_count += 1
        if source:
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
                threshold = self.min_fac * np.abs(np.amin(image))
            print("Threshold:", threshold)
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
            if cfg.profiling:
                time_start = time.time()
                self.profiler_count += 1
            retval = self.get_image_impl(idx).maxima
            if cfg.profiling:
                self.profiler_time += time.time() - time_start
            return retval
        except Exception as e:
            cfg.set_error(e, "Error returning coordinates "+str(e))
