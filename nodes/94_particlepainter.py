from node import *

def create():
    return ParticlePainterNode()


class ParticlePainterNode(Node):
    title = "Particle painter"
    group = "Reconstruction"
    colour = (143 / 255, 143 / 255, 143 / 255, 1.0)

    def __init__(self):
        super().__init__()  # Was: super(LoadDataNode, self).__init__()
        self.size = 270

        self.reconstruction_in = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.INPUT, parent = self)
        self.colour_out = ConnectableAttribute(ConnectableAttribute.TYPE_COLOUR, ConnectableAttribute.OUTPUT, parent = self)

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
