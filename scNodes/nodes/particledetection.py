from scNodes.core.node import *
from skimage.feature import peak_local_max

def create():
    return ParticleDetectionNode()


class ParticleDetectionNode(Node):
    description = "This node takes a Dataset as the input and outputs 'Coordinates'. These coordinates correspond to\n" \
                  "the location of local maxima in the input frames. The Particle Detection node is usually a precur-\n" \
                  "sor of a PSF Fitting node, which requires coordinates as the input and performs fitting at these\n" \
                  "coordinates."
    title = "Particle detection"
    group = "PSF-fitting reconstruction"
    colour = (230 / 255, 174 / 255, 13 / 255, 1.0)
    sortid = 1000

    METHODS = ["Local maximum"]
    THRESHOLD_OPTIONS = ["Value", "St. Dev.", "Mean", "Max", "Min"]

    def __init__(self):
        super().__init__()
        self.size = 290

        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["localizations_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.OUTPUT, parent=self)


        self.params["method"] = 0
        self.params["thresholding"] = 1
        self.params["threshold"] = 100
        self.params["sigmas"] = 2.0
        self.params["means"] = 3.0
        self.params["n_max"] = 2500
        self.params["d_min"] = 1
        self.params["max_fac"] = 0.75
        self.params["min_fac"] = 5.0

        self.threshold_value = -1

        self.active_roi = [0, 0, 1, 1]

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["localizations_out"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            self.connectable_attributes["localizations_out"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _c, self.use_roi = imgui.checkbox("use ROI", self.use_roi)
            self.any_change = self.any_change or _c
            imgui.push_item_width(160)
            #_c, self.params["method"] = imgui.combo("Detection method", self.params["method"], ParticleDetectionNode.METHODS)
            #self.any_change = self.any_change or _c
            _c, self.params["thresholding"] = imgui.combo("Threshold method", self.params["thresholding"], ParticleDetectionNode.THRESHOLD_OPTIONS)
            self.any_change = self.any_change or _c

            _c, self.params["n_max"] = imgui.slider_int("Max particles", self.params["n_max"], 100, 5000)
            Node.tooltip("Maximum amount of particles output per frame. Ctrl + click to override the slider\n"
                         "and enter a custom value.")
            self.any_change = self.any_change or _c

            imgui.pop_item_width()
            imgui.push_item_width(100)
            if self.params["thresholding"] == 0:
                _c, self.params["threshold"] = imgui.input_int("Value", self.params["threshold"], 0, 0)
                self.any_change = self.any_change or _c
            elif self.params["thresholding"] == 1:
                _c, self.params["sigmas"] = imgui.slider_float("x Sigma", self.params["sigmas"], 1.0, 5.0, format = "%.2f")
                self.any_change = self.any_change or _c
            elif self.params["thresholding"] == 2:
                _c, self.params["means"] = imgui.slider_float("x Mean", self.params["means"], 0.1, 10.0, format = "%.2f")
                self.any_change = self.any_change or _c
            elif self.params["thresholding"] == 3:
                _c, self.params["max_fac"] = imgui.slider_float("x Max", self.params["max_fac"], 0.0, 1.0, format="%.2f")
                self.any_change = self.any_change or _c
            elif self.params["thresholding"] == 4:
                _c, self.params["min_fac"] = imgui.slider_float("x Min", self.params["min_fac"], 1.0, 10.0, format="%.2f")
                self.any_change = self.any_change or _c
            Node.tooltip("The final threshold value is determined by this factor x the metric (sigma, mean, etc.)\n"
                         "chosen above. In case of Threshold method 'Value', the value entered here is the final \n"
                         "threshold value. Note that the metric is calculated for the ROI only.")
            if self.params["thresholding"] != 0:
                imgui.same_line()
                imgui.text(f" = {self.threshold_value:.1f}")



            _c, self.params["d_min"] = imgui.input_int("Minimum distance (px)", self.params["d_min"], 1, 10)
            self.any_change = self.any_change or _c
            self.params["d_min"] = max([1, self.params["d_min"]])
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        source = self.connectable_attributes["dataset_in"].get_incoming_node()
        if cfg.profiling and self.FRAME_REQUESTED_BY_IMAGE_VIEWER:
            self.profiler_count += 1
        if source:
            # Find threshold value
            image_obj = source.get_image(idx)
            image = image_obj.load()
            if self.use_roi:
                image = image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
            threshold = self.params["threshold"]
            if self.params["thresholding"] == 1:
                threshold = self.params["sigmas"] * np.std(image)
            elif self.params["thresholding"] == 2:
                threshold = self.params["means"] * np.mean(image)
            elif self.params["thresholding"] == 3:
                threshold = self.params["max_fac"] * np.amax(image)
            elif self.params["thresholding"] == 4:
                threshold = self.params["min_fac"] * np.abs(np.amin(image))
            self.threshold_value = threshold
            # Perform requested detection method
            coordinates = peak_local_max(image, threshold_abs = threshold, num_peaks = self.params["n_max"], min_distance = self.params["d_min"])
            values = image[coordinates[:, 0], coordinates[:, 1]]
            if self.use_roi:
                coordinates += np.asarray([self.roi[1], self.roi[0]])
                Node.get_source_load_data_node(self).dataset.reconstruction_roi = self.roi
                self.active_roi = self.roi
            else:
                Node.get_source_load_data_node(self).dataset.reconstruction_roi = [0, 0, image.shape[0], image.shape[1]]
                self.active_roi = [0, 0, image.shape[1], image.shape[0]]
            image_obj.maxima = coordinates
            image_obj.maxima_values = values
            return image_obj

    def get_roi(self):
        return self.active_roi

    def get_coordinates(self, idx=None):
        try:
            if cfg.profiling:
                time_start = time.time()
                self.profiler_count += 1

            out_image = self.get_image_impl(idx)
            retval = out_image.maxima, out_image.maxima_values
            if cfg.profiling:
                self.profiler_time += time.time() - time_start
            return retval
        except Exception as e:
            cfg.set_error(e, "Error returning coordinates "+str(e))

