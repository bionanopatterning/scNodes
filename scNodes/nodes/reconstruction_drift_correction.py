from scNodes.core.node import *
from scNodes.nodes.rcc import rcc

def create():
    return RCCNode()


class RCCNode(Node):
    description = "Correct sample drift based on the reconstructed particle coordinates, using the redundant cross correlation\n" \
                  "method by Wang et al. (2014) Optics Express (DOI: 10.1364/OE.22.015982). This operation can be computation-\n" \
                  "ally expensive, so the node makes a copy of the input Reconstruction and does not apply the correction on \n" \
                  "the fly; rather, it only applies (and updates its reconstruction data!) RCC when the user requests it."
    title = "Drift correction"
    group = "PSF-fitting reconstruction"
    colour = (243 / 255, 0 / 255, 80 / 255, 1.0)
    sortid = 1004
    enabled = True

    def __init__(self):
        super().__init__()
        self.size = 240

        self.connectable_attributes["reconstruction_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["reconstruction_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.OUTPUT, parent=self)

        self.params["temporal_bins"] = 5
        self.params["pixel_size"] = 100.0
        self.params["relative_to_idx"] = 0

        self.current_shifts_valid = False
        self.frame_x_shift = None
        self.frame_y_shift = None
        self.particle_dx = None
        self.particle_dy = None
        self.returns_image = False

    def render(self):
        if super().render_start():
            self.connectable_attributes["reconstruction_in"].render_start()
            self.connectable_attributes["reconstruction_out"].render_start()
            self.connectable_attributes["reconstruction_in"].render_end()
            self.connectable_attributes["reconstruction_out"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.set_next_item_width(50)
            _c, self.params["temporal_bins"] = imgui.input_int("Temporal bins", self.params["temporal_bins"], 0, 0)
            self.params["temporal_bins"] = max([2, self.params["temporal_bins"]])
            self.tooltip("Redundant cross correlation* works by rendering multiple, temporally distinct, super-resolution  \n"
                         "images. The 'Temporal bins' parameter specified how many sub images are used. For example, if   \n"
                         "there are 1000 frames in the dataset and the number of bins is 5, the subsets are frames 0-199, \n"
                         "200-399, etc. If the drift is predictable, a low number of bins can be sufficient. For unpre-\n"
                         "dictable drift, such as typically seen in cryo-stages, a larger number of bins can be better.\n"
                         "Using more bins slows down the computation.\n"
                         "*: Wang et al. (2014) Optics Express (DOI: 10.1364/OE.22.015982).")
            imgui.set_next_item_width(50)
            _c, self.params["pixel_size"] = imgui.input_float("Pixel size (nm)", self.params["pixel_size"], 0, 0, format='%.2f')
            self.tooltip("Redundant cross correlation* works by rendering multiple, temporally distinct, super-resolution  \n"
                         "images. The 'Pixel size' is the pixel size with which these images are rendered. Setting a lar-\n"
                         "ger value speeds up the computation, but excessively large values can negatively affect the re-"
                         "sult.\n"
                         "*: Wang et al. (2014) Optics Express (DOI: 10.1364/OE.22.015982).")
            imgui.set_next_item_width(50)
            _c, self.params["relative_to_idx"] = imgui.input_int("Home frame", self.params["relative_to_idx"], 0, 0)
            self.tooltip("The 'Home frame' is the frame relative to which the drift is corrected; i.e., the frame which is\n"
                         "considered to have 0 drift. When correlating the super-resolution reconstruction with other data\n"
                         "such as EM images or a second coloru channel, a home frame can be used to relate different coor-\n"
                         "dinate systems. E.g., by aligning the fluorescence timelapse to a bright field image, and align-\n"
                         "ing the EM to the bright field image as well. In such a case, you would want to set the home    \n"
                         "frame to be that fluorescence frame that was best aligned with the brightfield image. Typically,\n"
                         "this means to use a fluorescence image that was acquired immediately after that brightfield image.\n ")

            _cw = imgui.get_content_region_available_width()
            imgui.spacing()
            imgui.new_line()
            imgui.same_line(spacing=_cw / 2 - 55 / 2)
            if imgui.button("Run", 55, 25):
                self.do_drift_correction()

            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()

    def render_advanced(self):
        if self.current_shifts_valid:
            _cw = imgui.get_content_region_available_width()
            imgui.new_line()
            imgui.same_line(spacing=_cw / 2 - 100 / 2)
            if imgui.button("Plot drift", 100, 20):
                self.plot_drift()

    def on_update(self):
        if self.connectable_attributes["reconstruction_in"].newly_connected:
            self.current_shifts_valid = False
            self.connectable_attributes["reconstruction_in"].newly_connected = False
        if not self.connectable_attributes["reconstruction_in"].is_connected:
            self.current_shifts_valid = False

    def get_particle_data_impl(self):
        pdata = self.connectable_attributes["reconstruction_in"].get_incoming_node().get_particle_data()
        if self.current_shifts_valid:
            if len(pdata.parameters["x [nm]"]) == len(self.particle_dx):
                pdata.parameters["dx [nm]"] = self.particle_dx
                pdata.parameters["dy [nm]"] = self.particle_dy
                pdata.baked_by_renderer = False
        return pdata

    def do_drift_correction(self):
        try:
            datasource = self.connectable_attributes["reconstruction_in"].get_incoming_node()
            if datasource:
                pdata = datasource.get_particle_data()
                drift = rcc.rcc(pdata, self.params["temporal_bins"], self.params["pixel_size"])
                self.frame_x_shift = drift[0]
                self.frame_y_shift = drift[1]
                frame_idx = pdata.parameters["frame"].astype(int) - 1
                if self.params["relative_to_idx"] in frame_idx:
                    self.frame_x_shift -= self.frame_x_shift[self.params["relative_to_idx"]]
                    self.frame_y_shift -= self.frame_y_shift[self.params["relative_to_idx"]]
                self.particle_dx = self.frame_x_shift[frame_idx]
                self.particle_dy = self.frame_y_shift[frame_idx]
                self.current_shifts_valid = True
        except Exception as e:
            cfg.set_error(e, "Drift correction encountered an error:")

    def plot_drift(self):
        plt.plot(self.frame_x_shift, label="x drift (nm)", color=(0.0, 0.0, 0.5), linewidth=2)
        plt.plot(self.frame_y_shift, label="y drift (nm)", color=(0.5, 0.0, 0.0), linewidth=2)
        plt.title("Drift measured by redundant cross correlation")
        plt.legend()
        plt.ylabel("Drift (nm)")
        plt.xlabel("Frame nr.")
        plt.show()
