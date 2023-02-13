from scNodes.core.node import *
from scNodes.core import particlefitting as pfit

def create():
    return ParticleFittingNode()


class ParticleFittingNode(Node):
    title = "PSF fitting"
    group = "PSF-fitting reconstruction"
    colour = (230 / 255, 98 / 255, 13 / 255, 1.0)
    size = 300
    sortid = 1001
    RANGE_OPTIONS = ["All frames", "Current frame only", "Custom range", "Random subset"]
    ESTIMATORS = ["Least squares (GPU)", "Maximum likelihood (GPU)", "No estimator (CPU)"]
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
        self.params["psf"] = 0
        self.params["crop_radius"] = 3
        self.params["initial_sigma"] = 1.6
        self.fitting = False
        self.n_to_fit = 1
        self.n_fitted = 0
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

            if not self.particle_data.empty:
                imgui.text("Reconstruction info")
                imgui.text(f"particles: {self.particle_data.n_particles}")
                if not self.fitting:
                    imgui.text(f"x range: {self.particle_data.x_min / 1000.0:.1f} to {self.particle_data.x_max / 1000.0:.1f} um")
                    imgui.text(f"y range: {self.particle_data.y_min / 1000.0:.1f} to {self.particle_data.y_max / 1000.0:.1f} um")
            imgui.spacing()
            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()

    def render_advanced(self):
        imgui.set_next_item_width(80)
        _c, self.params["photons_per_count"] = imgui.input_float("photons per count", self.params["photons_per_count"], 0, 0, "%.2f")
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

    def init_fit(self):
        try:
            self.time_start = time.time()
            dataset_source = Node.get_source_load_data_node(self)
            self.particle_data = ParticleData(dataset_source.pixel_size)
            self.particle_data.set_reconstruction_roi(np.asarray(dataset_source.dataset.reconstruction_roi) * dataset_source.pixel_size)
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
        except Exception as e:
            self.fitting = False
            cfg.set_error(e, "Error in init_fit: "+str(e))

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
            cfg.set_error(e, "Error while fitting with PSF fitting node: "+str(e))

    def get_image_impl(self, idx=None):
        data_source = self.connectable_attributes["dataset_in"].get_incoming_node()
        coord_source = self.connectable_attributes["localizations_in"].get_incoming_node()
        if data_source and coord_source:
            frame = data_source.get_image(idx)
            coordinates = coord_source.get_coordinates(idx)
            frame.maxima = coordinates
            if frame.discard:
                frame.particles = None
                return frame
            particles = list()
            if self.params["estimator"] in [0, 1]:
                if self.params["psf"] == 0:
                    particles = pfit.frame_to_particles(frame, self.params["initial_sigma"], self.params["estimator"], self.params["crop_radius"],
                                                        constraints=[self.params["intensity_min"],
                                                                     self.params["intensity_max"], -1, -1, -1, -1,
                                                                     self.params["sigma_min"],
                                                                     self.params["sigma_max"],
                                                                     self.params["offset_min"],
                                                                     self.params["offset_max"]])
                elif self.params["psf"] == 1:
                    particles = pfit.frame_to_particles_3d(frame, self.params["initial_sigma"], self.params["estimator"], self.params["crop_radius"],
                                                        constraints=[self.params["intensity_min"],
                                                                     self.params["intensity_max"], -1, -1, -1, -1,
                                                                     self.params["sigma_min"],
                                                                     self.params["sigma_max"],
                                                                     self.params["sigma_min"],
                                                                     self.params["sigma_max"],
                                                                     self.params["offset_min"],
                                                                     self.params["offset_max"]])
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


