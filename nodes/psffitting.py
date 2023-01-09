from node import *
import particlefitting as pfit


def create():
    return ParticleFittingNode()


class ParticleFittingNode(Node):
    title = "PSF fitting"
    group = "PSF-fitting reconstruction"
    colour = (230 / 255, 98 / 255, 13 / 255, 1.0)
    size = 300
    sortid = 1001
    RANGE_OPTIONS = ["All frames", "Current frame only", "Custom range"]
    ESTIMATORS = ["Least squares (GPU)", "Maximum likelihood (GPU)", "No estimator (CPU)"]
    PSFS = ["Gaussian", "Elliptical Gaussian"]

    def __init__(self):
        super().__init__()

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.localizations_in = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.INPUT, parent=self)
        self.reconstruction_out = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.OUTPUT, parent=self)

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
            if self.range_option == 2:
                imgui.push_item_width(80)
                _c, (self.range_min, self.range_max) = imgui.input_int2('[start, stop) index', self.range_min,
                                                                        self.range_max)
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
                self.particle_data.bake()

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
            _roi = dataset_source.dataset.reconstruction_roi
            self.particle_data.set_reconstruction_roi(np.asarray([_roi[1], _roi[0]]) * dataset_source.pixel_size) ## 230109 check todo
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
            cfg.set_error(e, "Error in init_fit: "+str(e))

    def on_update(self):
        try:
            if self.fitting:
                if len(self.frames_to_fit) == 0:
                    self.fitting = False
                    self.play = False
                    self.particle_data.bake()
                    print(self.particle_data)
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
            cfg.set_error(e, "Error in ParticleFitNode.on_update(self): "+str(e))

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
                    x = frame.maxima[i, 1]
                    y = frame.maxima[i, 0]
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
        pass # cfg.pickle_temp["particle_data"] = self.particle_data
        # self.particle_data = ParticleData()

    def post_save_impl(self):
        pass # self.particle_data = cfg.pickle_temp["particle_data"]


