from node import *


def create():
    return ParticleFilterNode()


class ParticleFilterNode(Node):
    title = "Particle filter"
    group = "PSF-fitting reconstruction"
    colour = (230 / 255, 13 / 255, 13 / 255, 1.0)
    size = 310
    sortid = 1002
    def __init__(self):
        super().__init__()

        self.reconstruction_in = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.INPUT, parent=self)
        self.reconstruction_out = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.OUTPUT, parent=self)

        self.available_parameters = ["No data available"]
        self.filters = list()

        self.returns_image = False
        self.does_profiling_count = False

    def render(self):
        if super().render_start():
            self.reconstruction_in.render_start()
            self.reconstruction_out.render_start()
            self.reconstruction_in.render_end()
            self.reconstruction_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if self.reconstruction_in.check_connect_event():
                self.get_histogram_parameters()

            i = 0
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, 0.2, 0.2, 0.2, 1.0)
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, 0.2, 0.2, 0.2, 1.0)
            for pf in self.filters:
                imgui.push_id(f"pfid{i}")
                if imgui.button("-", 20, 20):
                    self.filters.remove(pf)
                    imgui.pop_id()
                    continue
                imgui.same_line()
                imgui.set_next_item_width(144)
                _c, pf.parameter = imgui.combo("Criterion", pf.parameter, self.available_parameters)
                if _c:
                    pf.set_data(*self.get_histogram_vals(self.available_parameters[pf.parameter]), self.available_parameters[pf.parameter])
                imgui.same_line(spacing = 10)
                _c, pf.invert = imgui.checkbox("NOT", pf.invert)
                Node.tooltip("If NOT is selected, the filter logic is inverted. Default\n"
                             "behaviour is: particle retained if min < parameter < max.\n"
                             "With NOT on, a particle is retained if max < parameter OR\nmin > parameter")

                pf.render()
                imgui.pop_id()
                i += 1
            content_width = imgui.get_window_content_region_width()
            imgui.new_line()
            imgui.same_line(content_width / 2 - 50)
            if imgui.button("Add filter", width = 100, height = 20):
                new_filter = ParticleFilterNode.Filter(self)
                new_filter.set_data(*self.get_histogram_vals(self.available_parameters[0]), self.available_parameters[0])
                self.filters.append(new_filter)
            imgui.pop_style_color(2)
            super().render_end()

    def on_gain_focus(self):
        self.get_histogram_parameters()

    def get_histogram_vals(self, parameter):
        datasource = self.reconstruction_in.get_incoming_node()
        if datasource:
            particledata = datasource.get_particle_data()
            return particledata.histogram_counts[parameter], particledata.histogram_bins[parameter]
        else:
            return np.asarray([0, 0]).astype('float32'), np.asarray([0, 1]).astype('float32')

    def get_histogram_parameters(self):
        datasource = self.reconstruction_in.get_incoming_node()
        if datasource:
            particledata = datasource.get_particle_data()
            self.available_parameters = list(particledata.histogram_counts.keys())

    def get_particle_data_impl(self):
        if cfg.profiling:
            time_start = time.time()
        datasource = self.reconstruction_in.get_incoming_node()
        if datasource:
            pdata = datasource.get_particle_data()
            for pf in self.filters:
                pf.apply(pdata)
            if cfg.profiling:
                self.profiler_time += time.time() - time_start
            return pdata
        else:
            if cfg.profiling:
                self.profiler_time += time.time() - time_start
            return ParticleData()

    class Filter:
        def __init__(self, parent):
            self.parameter_key = ""
            self.vals = [0, 0]
            self.bins = [0, 0]
            self.min = 0
            self.max = 1
            self.parameter = 0
            self.invert = False
            self.parent = parent

        def set_data(self, vals, bins, prm_key):
            self.vals = vals
            self.bins = bins
            self.min = bins[0]
            self.max = bins[-1]
            self.parameter_key = prm_key

        def render(self):
            content_width = imgui.get_content_region_available_width()
            imgui.plot_histogram("##histogram", self.vals, graph_size = (content_width, 40))
            imgui.text("{:.2f}".format(self.bins[0]))
            imgui.same_line(position=content_width - imgui.get_font_size() * len(str(self.bins[-1])) / 2)
            imgui.text("{:.2f}".format(self.bins[-1]))
            imgui.push_item_width(content_width)
            _c, self.min = imgui.slider_float("##min", self.min, self.bins[0], self.bins[-1], "min: %1.2f")
            _c, self.max = imgui.slider_float("##max", self.max, self.bins[0], self.bins[-1], "max: %1.2f")
            imgui.pop_item_width()
            imgui.separator()
            imgui.spacing()

        def apply(self, particle_data_object):
            if cfg.profiling:
                time_start = time.time()
            particle_data_object.apply_filter(self.parameter_key, self.min, self.max, logic_not=self.invert)
            if cfg.profiling:
                self.parent.profiler_time += time.time() - time_start
