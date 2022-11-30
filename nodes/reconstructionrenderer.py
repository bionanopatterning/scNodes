from node import *
from reconstruction import *


def create():
    return ReconstructionRendererNode()


class ReconstructionRendererNode(Node):
    title = "Render reconstruction"
    group = "Reconstruction"
    colour = (243 / 255, 0 / 255, 80 / 255, 1.0)
    size = 250
    COLOUR_MODE = ["RGB, LUT"]
    sortid = 1005
    def __init__(self):
        super().__init__()

        self.reconstruction_in = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.INPUT, parent = self)
        self.image_out = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.OUTPUT, parent = self)

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

        self.auto_render = False

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

            if self.auto_render:
                if self.any_change:
                    self.build_reconstruction()
            else:
                _cw = imgui.get_content_region_available_width()
                imgui.new_line()
                imgui.same_line(spacing = _cw / 2 - 70 / 2)
                if imgui.button("Render", 70, 30):
                    self.build_reconstruction()


            if _mag_changed:
                self.original_pixel_size = Node.get_source_load_data_node(self).dataset.pixel_size
                roi = self.get_particle_data().reconstruction_roi
                img_width = int((roi[3] - roi[1]) * self.magnification)
                img_height = int((roi[2] - roi[0]) * self.magnification)
                self.reconstruction_image_size = (img_width, img_height)

            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()

    def render_advanced(self):
        _c, self.auto_render = imgui.checkbox("Auto render", self.auto_render)
        self.tooltip("When checked, instead of displaying a 'render' button,\n"
                     "the node will always render the reconstruction upon any\n"
                     "change to the settings.")

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
            cfg.set_error(e, "Error building reconstruction.\n"+str(e))

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
        cfg.pickle_temp["latest_image"] = self.latest_image
        self.latest_image = None

    def post_save_impl(self):
        self.latest_image = cfg.pickle_temp["latest_image"]

