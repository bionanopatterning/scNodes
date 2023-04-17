from scNodes.core.node import *
from scNodes.core.opengl_classes import Texture

def create():
    return DiagnosticPlotsNode()


class DiagnosticPlotsNode(Node):
    description = "Takes a reconstruction as the input and generates plots, which are shown in a pop-up window.\n" \
                  "The node does not output images; instead, plots can be saved and interacted with in the pop-\n" \
                  "up. There are two plotting modes:\n" \
                  "1) Histogram: plot histograms of particle parameters, e.g. the uncertainty.\n" \
                  "2) Scatter plot: generate scatter plots that relate two (or three, via colour) parameters.\n" \
                  "\tFor example, plotting 'frame' versus 'intensity', with dots coloured by 'uncertainty'\n" \
                  "\tcan help visualize whether and how the images change over time (but can also be hard\n" \
                  "\tto interpret)."
    title = "Diagnostic plots"
    group = "PSF-fitting reconstruction"
    colour = (143 / 255, 143 / 255, 143 / 255, 1.0)
    size = 250
    sortid = 1006

    PLOT_TYPES = ["Histogram", "Scatter plot"]

    def __init__(self):
        super().__init__()

        self.connectable_attributes["reconstruction_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.INPUT, parent=self)

        self.available_parameters = list()
        self.params["x_param"] = 0
        self.params["y_param"] = 0
        self.params["colourize"] = False
        self.params["alpha"] = 1.0
        self.params["c_param"] = 0
        self.params["cmin"] = 0.0
        self.params["cmax"] = 0.0
        self.params["n_bins"] = -1
        self.params["plot_type"] = 0
        self.histogram_bins = [0, 0]
        self.histogram_values = [0, 0]

        self.lut_texture = Texture(format="rgb32f")
        self.params["lut"] = 0
        self.cmap = np.zeros((256, 3))

    def render(self):
        if super().render_start():
            self.connectable_attributes["reconstruction_in"].render_start()
            self.connectable_attributes["reconstruction_in"].render_end()

            if self.connectable_attributes["reconstruction_in"].check_connect_event() or self.connectable_attributes["reconstruction_in"].check_disconnect_event():
                self.on_gain_focus()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.set_next_item_width(170)
            _, self.params["plot_type"] = imgui.combo("Plot type", self.params["plot_type"], DiagnosticPlotsNode.PLOT_TYPES)
            imgui.set_next_item_width(170)
            _, self.params["x_param"] = imgui.combo("X axis", self.params["x_param"], self.available_parameters)
            if self.params["plot_type"] == 0:
                imgui.set_next_item_width(60)
                _, self.params["n_bins"] = imgui.input_int("# bins", self.params["n_bins"], 0, 0)
                self.tooltip("Use -1 to automatically determine the number of bins.")
            if self.params["plot_type"] == 1:
                imgui.set_next_item_width(170)
                _, self.params["y_param"] = imgui.combo("Y axis", self.params["y_param"], self.available_parameters)
                # colourize:
                imgui.set_next_item_width(170)
                _c, self.params["colourize"] = imgui.checkbox("Colourize", self.params["colourize"])
                if _c:
                    self.update_lut()
                    self.get_histogram_values()
                    self.params["cmin"] = float(self.histogram_bins[0])
                    self.params["cmax"] = float(self.histogram_bins[-1])
                if self.params["colourize"]:
                    imgui.set_next_item_width(170)
                    _c, self.params["c_param"] = imgui.combo("Colour by", self.params["c_param"], self.available_parameters)
                    if _c:
                        self.get_histogram_values()
                        self.params["cmin"] = float(self.histogram_bins[0])
                        self.params["cmax"] = float(self.histogram_bins[-1])
                    imgui.set_next_item_width(170)
                    _c, self.params["lut"] = imgui.combo("LUT", self.params["lut"], settings.lut_names)
                    if _c:
                        self.update_lut()
                    _cw = imgui.get_content_region_available_width()
                    imgui.plot_histogram("##hist", self.histogram_values, graph_size=(_cw, 50.0))
                    _l = self.histogram_bins[0]
                    _h = self.histogram_bins[-1]
                    _max = self.params["cmax"]
                    _min = self.params["cmin"]
                    _uv_left = 0.5
                    _uv_right = 0.5
                    if _max != _min:
                        _uv_left = 1.0 + (_l - _max) / (_max - _min)
                        _uv_right = 1.0 + (_h - _max) / (_max - _min)
                    imgui.image(self.lut_texture.renderer_id, _cw, 10.0, (_uv_left, 0.5), (_uv_right, 0.5), border_color=(0.0, 0.0, 0.0, 1.0))
                    imgui.push_item_width(_cw)
                    _c, self.params["cmin"] = imgui.slider_float("##min", self.params["cmin"], self.histogram_bins[0], self.histogram_bins[1], format='min: %.1f')
                    _c, self.params["cmax"] = imgui.slider_float("##max", self.params["cmax"], self.histogram_bins[0], self.histogram_bins[1], format='max: %.1f')
                    _c, self.params["alpha"] = imgui.slider_float("##alpha", self.params["alpha"], 0.0, 1.0, format="alpha: %.1f")
                    imgui.pop_item_width()

            _cw = imgui.get_content_region_available_width()
            imgui.new_line()
            imgui.same_line(spacing=_cw / 2 - 70 / 2)
            if imgui.button("Render", 70, 30):
                self.make_and_show_plot()

            super().render_end()

    def get_histogram_values(self):
        datasource = self.connectable_attributes["reconstruction_in"].get_incoming_node()
        if datasource:
            particledata = datasource.get_particle_data()
            self.available_parameters = list(particledata.histogram_bins.keys())
            if self.available_parameters[self.params["c_param"]] in list(particledata.histogram_counts.keys()):
                self.histogram_values = particledata.histogram_counts[self.available_parameters[self.params["c_param"]]]
                self.histogram_bins = particledata.histogram_bins[self.available_parameters[self.params["c_param"]]]

    def make_and_show_plot(self):
        try:
            particledata = self.connectable_attributes["reconstruction_in"].get_incoming_node().get_particle_data()
            if self.params["plot_type"] == 0:
                # plot histogram
                bins = None if self.params["n_bins"] == -1 else self.params["n_bins"]
                data = particledata.parameters[self.available_parameters[self.params["x_param"]]]
                plt.hist(data, bins=bins, color = (0.0, 0.0, 0.0, 1.0))
                plt.xlabel(self.available_parameters[self.params["x_param"]])
                plt.title(f"Histogram of particle {self.available_parameters[self.params['x_param']]}")
            elif self.params["plot_type"] == 1:
                # plot scatter plot
                xdata = particledata.parameters[self.available_parameters[self.params["x_param"]]]
                ydata = particledata.parameters[self.available_parameters[self.params["y_param"]]]
                # colour
                if self.params["colourize"]:
                    cdata = particledata.parameters[self.available_parameters[self.params["c_param"]]]
                    n_particles = particledata.n_particles
                    lut_len = self.cmap.shape[1]
                    colour = np.zeros((n_particles, 4))
                    _min = self.params["cmin"]
                    _denom = self.params["cmax"] - _min
                    for i in range(n_particles):
                        c_idx = (cdata[i] - _min) / (_denom)
                        c_idx = int(c_idx * lut_len)
                        c_idx = np.max([np.min([lut_len - 1, c_idx]), 0])
                        colour[i, :3] = self.cmap[0, c_idx, :]
                    colour[:, 3] = self.params["alpha"]
                    plt.scatter(xdata, ydata, s=3, color=colour)
                else:
                    plt.scatter(xdata, ydata, s=3, color=(0.0, 0.0, 0.0, self.params["alpha"]))
                plt.xlabel(self.available_parameters[self.params["x_param"]])
                plt.ylabel(self.available_parameters[self.params["y_param"]])
                if self.params["colourize"]:
                    plt.title(f'Scatter plot of particle {self.available_parameters[self.params["x_param"]]} vs {self.available_parameters[self.params["y_param"]]}, coloured by {self.available_parameters[self.params["c_param"]]}')
                else:
                    plt.title(f'Scatter plot of particle {self.available_parameters[self.params["x_param"]]} vs {self.available_parameters[self.params["y_param"]]}')
            plt.show()
        except Exception as e:
            cfg.set_error(e, "Error in DiagnosticPlotNode:")

    def on_gain_focus(self):
        datasource = self.connectable_attributes["reconstruction_in"].get_incoming_node()
        if datasource:
            particledata = datasource.get_particle_data()
            if particledata:
                if particledata.baked:
                    self.available_parameters = list(particledata.histogram_counts.keys())

    def update_lut(self):
        lut_array = np.asarray(settings.luts[settings.lut_names[self.params["lut"]]])
        if lut_array.shape[1] == 3:
            lut_array = np.reshape(lut_array, (1, lut_array.shape[0], 3))
            self.cmap = lut_array
            self.lut_texture.update(lut_array)