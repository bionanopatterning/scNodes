import imgui

from node import *
from reconstruction import *


def create():
    return ReconstructionRendererNode()


class ReconstructionRendererNode(Node):
    title = "Render reconstruction"
    group = "PSF-fitting reconstruction"
    colour = (243 / 255, 0 / 255, 80 / 255, 1.0)
    size = 250
    COLOUR_MODE = ["RGB, LUT"]
    sortid = 1005

    def __init__(self):
        super().__init__()

        self.reconstruction_in = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.INPUT, parent = self)
        self.image_out = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.OUTPUT, parent = self)

        self.pixel_size = 10.0
        self.default_sigma = 30.0
        self.fix_sigma = False

        self.reconstructor = Reconstructor()
        self.latest_image = None

        self.original_pixel_size = 100.0
        self.reconstruction_image_size = [1, 1]

        # painting particles
        self.paint_particles = False
        self.paint_by = 0
        self.paint_applied = False

        self.parameter = 0
        self.available_parameters = ["Fixed colour"]
        self.colour_min = (0.0, 0.0, 0.0)
        self.colour_max = (1.0, 1.0, 1.0)
        self.histogram_values = np.asarray([0, 0]).astype('float32')
        self.histogram_bins = np.asarray([0, 0]).astype('float32')
        self.histogram_initiated = False
        self.min = 0
        self.max = 0
        #
        self.does_profiling_time = False
        self.does_profiling_count = False

        self.auto_render = False
        self.output_mode = 0

    def render(self):
        if super().render_start():
            self.reconstruction_in.render_start()
            self.image_out.render_start()
            self.reconstruction_in.render_end()
            self.image_out.render_end()


            if self.reconstruction_in.check_connect_event() or self.reconstruction_in.check_disconnect_event():
                self.init_histogram_values()
                self.on_gain_focus()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.push_item_width(90)
            imgui.text("Pixel size:")
            imgui.same_line()
            _c, self.pixel_size = imgui.input_float("##Pixel size", self.pixel_size, 0.0, 0.0, format='%.2f (nm)')
            pxs_changed = _c
            self.any_change = _c or self.any_change
            imgui.text(f"Final image size: {self.reconstruction_image_size[0]} x {self.reconstruction_image_size[1]} px")

            # Colourize optionsx
            _c, self.paint_particles = imgui.checkbox("Colourize particles", self.paint_particles)
            if _c and not self.paint_particles:
                self.paint_by = 0
                self.reconstructor.set_lut(0)
            if self.paint_particles:
                _cw = imgui.get_content_region_available_width()
                imgui.push_item_width(_cw - 100)
                _c, self.paint_by = imgui.combo("LUT", self.paint_by, settings.lut_names)
                if _c:
                    self.reconstructor.set_lut(self.paint_by)
                    self.paint_applied = False
                _c, self.parameter = imgui.combo("Parameter", self.parameter, self.available_parameters)
                if _c:
                    self.get_histogram_values()
                    self.min = self.histogram_bins[0]
                    self.max = self.histogram_bins[-1]
                    self.paint_applied = False
                imgui.pop_item_width()
                # TODO: histogram

            if self.auto_render:
                if self.any_change:
                    self.build_reconstruction()
            else:
                _cw = imgui.get_content_region_available_width()
                imgui.new_line()
                imgui.same_line(spacing = _cw / 2 - 70 / 2)
                if imgui.button("Render", 70, 30):
                    self.build_reconstruction()

            if pxs_changed:
                roi = self.get_particle_data().reconstruction_roi
                img_width = int((roi[3] - roi[1]) / self.pixel_size)
                img_height = int((roi[2] - roi[0]) / self.pixel_size)
                self.reconstruction_image_size = (img_width, img_height)

            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()


    def render_advanced(self):
        _c, self.fix_sigma = imgui.checkbox("Force uncertainty", self.fix_sigma)

        if self.fix_sigma:
            imgui.same_line(spacing=10)
            imgui.push_item_width(50)
            _, self.default_sigma = imgui.input_float(" nm", self.default_sigma, 0, 0, format="%.1f")
            imgui.pop_item_width()

        _c, self.output_mode = imgui.combo("Output mode", self.output_mode, ["float", "uint16"])
        if _c:
            if self.output_mode == 0:
                self.reconstructor.set_mode("float")
            else:
                self.reconstructor.set_mode("ui16")

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
            if particledata:
                if particledata.baked:
                    self.available_parameters = ["Fixed colour"] + list(particledata.histogram_counts.keys())
                    self.parameter = min([1, len(self.available_parameters)])
                    self.histogram_values = particledata.histogram_counts[self.available_parameters[self.parameter]]
                    self.histogram_bins = particledata.histogram_bins[self.available_parameters[self.parameter]]
                    self.min = self.histogram_bins[0]
                    self.max = self.histogram_bins[-1]


    def apply_paint(self, particle_data):
        if cfg.profiling:
            time_start = time.time()
        values = particle_data.parameter[self.available_parameters[self.parameter]]
        i = 0
        for particle in particle_data.particles:
            fac = min([max([(values[i] - self.min) / self.max, 0.0]), 1.0])
            _colour_idx = fac
            particle.colour_idx = _colour_idx
            i += 1
        self.reconstructor.colours_set = False
        if cfg.profiling:
            self.profiler_time += time.time() - time_start

    def build_reconstruction(self):
        try:
            self.original_pixel_size = Node.get_source_load_data_node(self).dataset.pixel_size
            self.reconstructor.set_pixel_size(self.pixel_size)
            self.reconstructor.set_image_size(self.reconstruction_image_size)
            datasource = self.reconstruction_in.get_incoming_node()
            if datasource:
                particle_data = datasource.get_particle_data()
                self.reconstructor.set_particle_data(particle_data)
                self.reconstructor.set_camera_origin([-particle_data.reconstruction_roi[0], -particle_data.reconstruction_roi[1]])

                ## Apply colours
                if self.paint_particles:
                    if not self.paint_applied:
                        self.apply_paint(particle_data)
                        self.paint_applied = True
                else:
                    if not self.paint_applied:
                        for particle in particle_data.particles:
                            particle.colour_idx = 1.0
                        self.paint_applied = True
                if self.reconstructor.particle_data.empty:
                    return None
                else:
                    self.latest_image = self.reconstructor.render(fixed_uncertainty=(self.default_sigma if self.fix_sigma else None))
                    self.any_change = True
            else:
                self.latest_image = None
        except Exception as e:
            cfg.set_error(e, "Error building reconstruction.\n"+str(e))

    def get_image_impl(self, idx=None):
        if self.latest_image is not None:
            img_wrapper = Frame("super-resolution reconstruction virtual frame")
            img_wrapper.data = self.latest_image
            img_wrapper.pixel_size = self.pixel_size
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
        roi = None
        try:
            roi = self.get_particle_data().reconstruction_roi
        except Exception as e:
            cfg.set_error(e, f"Couldn't get particle data in {self.title} node.")
        if roi is not None:
            img_width = int((roi[2] - roi[0]) / self.pixel_size)  # TODO
            img_height = int((roi[3] - roi[1]) / self.pixel_size)
            self.reconstruction_image_size = (img_width, img_height)

    def pre_save_impl(self):
        cfg.pickle_temp["latest_image"] = self.latest_image
        self.latest_image = None

    def post_save_impl(self):
        self.latest_image = cfg.pickle_temp["latest_image"]

