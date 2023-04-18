from scNodes.core.node import *
from scNodes.core import particlefitting as pfit
from scNodes.core.util import tic, toc

def create():
    return ParticleFittingNode()


class ParticleFittingNode(Node):
    description = "Perform PSF fitting in the input frames at the location of the input coordinates. This node uses\n" \
                  "'pyGPUfit' for GPU-accelerated fitting. Unfortunately, pyGPUfit uses CUDA, which is not available\n" \
                  "on many macOS devices.\n" \
                  "\n" \
                  "The node outputs a 'Reconstruction', which is essentially just a list of particle positions,\n" \
                  "intensities, uncertainties, etc., and which, in scNodes, is a datatype that can be further\n" \
                  "processed by a number of other nodes that are also found in the 'PSF-fitting reconstruction'\n" \
                  "node group. "
    title = "PSF fitting"
    group = "PSF-fitting reconstruction"
    colour = (230 / 255, 98 / 255, 13 / 255, 1.0)
    size = 300
    sortid = 1001
    RANGE_OPTIONS = ["All frames", "Current frame only", "Custom range", "Random subset"]
    ESTIMATORS = ["Least squares (GPU)", "Maximum likelihood (GPU)", "No estimator (CPU)"]
    UNCERTAINTY_ESTIMATOR = ["Thompson", "Mortensen"]
    PSFS = ["Gaussian", "Elliptical Gaussian"]

    def __init__(self):
        super().__init__()

        # Set up connectable attributes
        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["localizations_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["reconstruction_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.OUTPUT, parent=self)

        self.params["range_option"] = 1
        self.params["range_min"] = 0
        self.params["range_max"] = 1
        self.params["subset_size"] = 50
        self.params["estimator"] = 1
        self.params["uncertainty_estimator"] = 1
        self.params["psf"] = 0
        self.params["crop_radius"] = 3
        self.params["initial_sigma"] = 1.6
        self.params["skip_discard"] = True
        self.fitting = False
        self.n_to_fit = 1
        self.n_fitted = 0
        self.n_frames_discarded = 0
        self.frames_to_fit = list()
        self.particle_data = ParticleData()

        self.time_start = 0
        self.time_stop = 0

        self.params["intensity_min"] = 100.0
        self.params["intensity_max"] = -1.0
        self.params["sigma_min"] = 1.0
        self.params["sigma_max"] = 10.0
        self.params["offset_min"] = 0.0
        self.params["offset_max"] = -1.0

        self.params["custom_bounds"] = False
        self.params["photons_per_count"] = 0.45
        self.params["camera_offset"] = 100.0
        self.detection_roi = [0, 0, 1, 1]

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["reconstruction_out"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            self.connectable_attributes["reconstruction_out"].render_end()
            imgui.new_line()
            self.connectable_attributes["localizations_in"].render_start()
            self.connectable_attributes["localizations_in"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.set_next_item_width(200)
            _c, self.params["estimator"] = imgui.combo("Estimator", self.params["estimator"], ParticleFittingNode.ESTIMATORS)
            imgui.set_next_item_width(148)
            if self.params["estimator"] != 2:
                _c, self.params["psf"] = imgui.combo("PSF", self.params["psf"], ParticleFittingNode.PSFS)
            if self.params["estimator"] in [0, 1]:
                imgui.same_line()
                imgui.button("?", 19, 19)
                self.tooltip("This node uses pyGpufit, a GPU fitting library by Przybylski et al. Original publication:\n"
                             "Przybylski et al. (2017) Gpufit: An open-source toolkit for GPU-accelerated curve fitting.\n"
                             "Sci. Rep. 7:15722. doi: 10.1038/s41598-017-15313-9")
            self.any_change = _c or self.any_change
            imgui.push_item_width(80)
            if self.params["estimator"] in [0, 1]:
                _c, self.params["initial_sigma"] = imgui.input_float("Initial sigma (px)", self.params["initial_sigma"], 0, 0, "%.1f")
                self.any_change = _c or self.any_change
                _c, self.params["crop_radius"] = imgui.input_int("Fitting radius (px)", self.params["crop_radius"], 0, 0)
                self.any_change = _c or self.any_change
            imgui.pop_item_width()
            imgui.spacing()
            imgui.set_next_item_width(180)
            _c, self.params["range_option"] = imgui.combo("Range", self.params["range_option"],
                                                ParticleFittingNode.RANGE_OPTIONS)
            if self.params["range_option"] == 2:
                imgui.push_item_width(80)
                _c, (self.params["range_min"], self.params["range_max"]) = imgui.input_int2('[start, stop) index', self.params["range_min"],
                                                                        self.params["range_max"])
                imgui.pop_item_width()
            if self.params["range_option"] == 3:
                imgui.push_item_width(80)
                _c, self.params["subset_size"] = imgui.input_int("# frames", self.params["subset_size"], 0, 0)
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
                self.particle_data.bake()
                self.any_change = True

            imgui.text("Reconstruction info:")
            if not self.particle_data.empty:
                imgui.text(f"\tparticles: {self.particle_data.n_particles}")
                imgui.text(f"\tframes discarded: {self.n_frames_discarded}")
                if not self.fitting:
                    imgui.text(f"\tx range: {self.particle_data.x_min / 1000.0:.1f} to {self.particle_data.x_max / 1000.0:.1f} um")
                    imgui.text(f"\ty range: {self.particle_data.y_min / 1000.0:.1f} to {self.particle_data.y_max / 1000.0:.1f} um")
            else:
                imgui.text("\tno particles detected")
                imgui.text(f"\tframes discarded: {self.n_frames_discarded}")
            imgui.spacing()
            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()

    def render_advanced(self):
        imgui.set_next_item_width(100)
        _c, self.params["uncertainty_estimator"] = imgui.combo("Uncertainty estimator", self.params["uncertainty_estimator"], ParticleFittingNode.UNCERTAINTY_ESTIMATOR)
        self.tooltip("Select which estimator for the localization precision to use.\n"
                     "Thompson et al.  2002: Precise Nanometer Localization Analysis for Individual Fluorescent Probes,              Biophysical Journal\n"
                     "Mortensen et al. 2010: Optimized localization analysis for single-molecule tracking and super-resolution microscopy,    Nat. Meth.\n"
                     "\n"
                     "Mortensen is faster and better (at least for MLE); Thompson was previously used in scNodes (and ThunderSTORM) and is a legacy mode.\n")
        if _c:
            self.fitting = False  # stop fitting if uncertainty estimator is changed during fitting.
        self.mark_change(_c)
        imgui.set_next_item_width(50)
        _c, self.params["photons_per_count"] = imgui.input_float("photons per count", self.params["photons_per_count"], 0, 0, "%.2f")
        self.tooltip("Specify how many digital counts are generated by one photon hitting the camera. ")
        imgui.set_next_item_width(50)
        _c, self.params["camera_offset"] = imgui.input_float("camera offset", self.params["camera_offset"], 0, 0, "%.2f")
        imgui.spacing()
        _c, self.params["custom_bounds"] = imgui.checkbox("Use custom parameter bounds", self.params["custom_bounds"])
        Node.tooltip("Edit the bounds for particle parameters intensity, sigma, and offset.")
        if self.params["custom_bounds"]:
            imgui.push_item_width(45)
            _c, self.params["sigma_min"] = imgui.input_float("min##sigma", self.params["sigma_min"], 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle sigma to a minimum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            _c, self.params["sigma_max"] = imgui.input_float("max##sigma", self.params["sigma_max"], 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle sigma to a maximum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            imgui.text("sigma (px)")

            _c, self.params["intensity_min"] = imgui.input_float("min##int", self.params["intensity_min"], 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle intensity to a minimum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            _c, self.params["intensity_max"] = imgui.input_float("max##int", self.params["intensity_max"], 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle intensity to a maximum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            imgui.text("intensity (counts)")

            _c, self.params["offset_min"] = imgui.input_float("min##offset", self.params["offset_min"], 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle offset to a minimum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            _c, self.params["offset_max"] = imgui.input_float("max##offset", self.params["offset_max"], 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle offset to a maximum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            imgui.text("offset (counts)")

            imgui.pop_item_width()

        _c, self.params["skip_discard"] = imgui.checkbox("Skip 'discarded' frames", self.params["skip_discard"])

    def init_fit(self):
        try:
            self.time_start = time.time()
            dataset_source = Node.get_source_load_data_node(self)
            self.particle_data = ParticleData(dataset_source.pixel_size)
            self.particle_data.photons_per_count = self.params["photons_per_count"]
            self.particle_data.camera_offset = self.params["camera_offset"]
            self.particle_data.uncertainty_estimator = self.params["uncertainty_estimator"]
            self.fitting = True
            self.frames_to_fit = list()
            dataset = dataset_source.dataset
            n_frames = dataset.n_frames
            if self.params["range_option"] == 0:
                self.frames_to_fit = list(range(0, n_frames))
            elif self.params["range_option"] == 2:
                self.frames_to_fit = list(range(self.params["range_min"], self.params["range_max"]))
            elif self.params["range_option"] == 1:
                self.frames_to_fit = [dataset.current_frame]
            elif self.params["range_option"] == 3:
                self.frames_to_fit = np.random.choice(n_frames, size=min([n_frames, self.params["subset_size"]]), replace=False).tolist()
            self.n_to_fit = len(self.frames_to_fit)
            self.n_fitted = 0
            self.n_frames_discarded = 0
        except Exception as e:
            self.fitting = False
            cfg.set_error(e, "Error in init_fit: "+str(e))

    def on_update(self):
        try:
            if self.fitting:
                if len(self.frames_to_fit) == 0:
                    self.fitting = False
                    self.play = False
                    self.particle_data.set_reconstruction_roi(np.asarray(self.detection_roi) * self.particle_data.pixel_size)
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
            cfg.set_error(e, "Error while fitting with PSF fitting node: "+str(e))

    def get_image_impl(self, idx=None):
        data_source = self.connectable_attributes["dataset_in"].get_incoming_node()
        coord_source = self.connectable_attributes["localizations_in"].get_incoming_node()
        if data_source and coord_source:
            frame = data_source.get_image(idx)
            if self.params["skip_discard"] and frame.discard:
                frame.particles = None
                self.n_frames_discarded += 1
                return frame
            coordinates = coord_source.get_coordinates(idx)
            self.detection_roi = coord_source.get_roi()
            frame.maxima = coordinates
            particles = list()
            if self.params["estimator"] in [0, 1]:
                if self.params["psf"] == 0:
                    particles = pfit.frame_to_particles(frame, self.params["initial_sigma"], self.params["estimator"], self.params["crop_radius"],
                                                        constraints=[self.params["intensity_min"],
                                                                     self.params["intensity_max"], -1, -1, -1, -1,
                                                                     self.params["sigma_min"],
                                                                     self.params["sigma_max"],
                                                                     self.params["offset_min"],
                                                                     self.params["offset_max"]],
                                                        uncertainty_estimator=self.params["uncertainty_estimator"],
                                                        camera_offset=self.params["camera_offset"])
                elif self.params["psf"] == 1:
                    particles = pfit.frame_to_particles_3d(frame, self.params["initial_sigma"], self.params["estimator"], self.params["crop_radius"],
                                                           constraints=[self.params["intensity_min"],
                                                                        self.params["intensity_max"], -1, -1, -1, -1,
                                                                        self.params["sigma_min"],
                                                                        self.params["sigma_max"],
                                                                        self.params["sigma_min"],
                                                                        self.params["sigma_max"],
                                                                        self.params["offset_min"],
                                                                        self.params["offset_max"]],
                                                           uncertainty_estimator=self.params["uncertainty_estimator"],
                                                           camera_offset=self.params["camera_offset"])
            elif self.params["estimator"] == 2:
                x = np.empty(len(frame.maxima))
                y = np.empty(len(frame.maxima))
                intensity = np.empty(len(frame.maxima))
                pxd = frame.load()
                for i in range(len(frame.maxima)):
                    intensity[i] = pxd[frame.maxima[i, 0], frame.maxima[i, 1]]
                    x[i] = frame.maxima[i, 1]
                    y[i] = frame.maxima[i, 0]
                particles = dict()
                particles["x [nm]"] = x
                particles["y [nm]"] = y
                particles["intensity [counts]"] = intensity
            frame.particles = particles
            new_maxima = list()
            new_maxima_x = frame.particles["x [nm]"]
            new_maxima_y = frame.particles["y [nm]"]
            for _x, _y in zip(new_maxima_x, new_maxima_y):
                new_maxima.append([_y, _x])
            frame.maxima = new_maxima
            return frame

    def get_particle_data_impl(self):
        self.particle_data.clean()
        return self.particle_data

    def pre_pickle_impl(self):
        cfg.pickle_temp["particle_data"] = self.particle_data
        self.particle_data = ParticleData()

    def post_pickle_impl(self):
        self.particle_data = cfg.pickle_temp["particle_data"]


