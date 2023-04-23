from scNodes.core.node import *
from scipy.interpolate import InterpolatedUnivariateSpline

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


# Redundant cross correlation (RCC):
# Wang et al.  'Localization events-based sample drift correction for localization microscopy with redundant cross-correlation algorithm' (2014), Optics Express. DOI: 10.1364/OE.22.015982

# Code partially based on the Picasso python implementation of RCC (at https://github.com/jungmannlab/picasso)
# by Schnitzbauer et al., in 'Super-resolution microscopy with DNA-PAINT' (2017) Nature Protocols. DOI: 10.1038/nprot.2017.024

def xcorr(f, g):
    F = np.fft.fft2(f)
    G_star = np.conj(np.fft.fft2(g))
    return np.fft.fftshift(np.real(np.fft.ifft2(F * G_star)))


def find_peak_in_xcorr(xcorr, fit_roi=5):
    if xcorr.shape[0] < (fit_roi * 2 + 1) or xcorr.shape[1] < (fit_roi * 2 + 1):
        raise Exception("fit_roi too large: fit_roi * 2 + 1 > xcorr shape.")
    x, y = np.unravel_index(np.argmax(xcorr), xcorr.shape)

    x_range = [x - fit_roi, x + fit_roi + 1]
    y_range = [y - fit_roi, y + fit_roi + 1]
    if x_range[0] < 0:
        x_range = [0, 2 * fit_roi + 1]
    elif x_range[1] > xcorr.shape[0]:
        x_range = [xcorr.shape[0] - 2 * fit_roi - 1, xcorr.shape[0]]
    if y_range[0] < 0:
        y_range = [0, 2 * fit_roi + 1]
    elif y_range[1] > xcorr.shape[1]:
        y_range = [xcorr.shape[1] - 2 * fit_roi - 1, xcorr.shape[1]]
    # crop:
    crop = xcorr[x_range[0]:x_range[1], y_range[0]:y_range[1]]
    # centroid:
    mass = 0
    moment_x = 0
    moment_y = 0
    for _x in range(fit_roi * 2 + 1):
        for _y in range(fit_roi * 2 + 1):
            m = crop[_x, _y]
            moment_x += _x * m
            moment_y += _y * m
            mass += m
    com_x = moment_x / mass
    com_y = moment_y / mass
    x = com_x + x_range[0] - xcorr.shape[0] // 2
    y = com_y + y_range[0] - xcorr.shape[1] // 2
    return [x, y]


def rcc(particle_data, segments=10, pixel_size=200.0):
    x = particle_data.parameters["x [nm]"]
    y = particle_data.parameters["y [nm]"]
    f = particle_data.parameters["frame"]

    x_min = np.amin(x)
    x_max = np.amax(x)
    y_min = np.amin(y)
    y_max = np.amax(y)
    x_range = [x_min, x_max - (x_max - x_min) % pixel_size]
    y_range = [y_min, y_max - (y_max - y_min) % pixel_size]
    n_bins_x = int((x_range[1] - x_range[0]) / pixel_size)
    n_bins_y = int((y_range[1] - y_range[0]) / pixel_size)
    n_frames = int(np.amax(f))

    starting_indices = (np.linspace(0, 1, segments+1) * n_frames).astype(int)
    images = list()
    for i in range(segments):
        start = starting_indices[i]
        stop = starting_indices[i+1]

        f_mask = (start <= f) * (f < stop)
        _x = x[f_mask]
        _y = y[f_mask]

        # generate an image
        img, _, _, _ = plt.hist2d(_x, _y, bins=[n_bins_x, n_bins_y], range=[x_range, y_range])
        plt.close()
        images.append(img)

    # correlate the images
    shifts = np.zeros((segments, segments, 2))
    for i in range(segments):
        for j in range(i+1, segments):
            # compute the image shift for these two segments
            corr = xcorr(images[i], images[j])
            shifts[i, j] = find_peak_in_xcorr(corr)

    # compute the drift for every segment - code below in particular based on Picasso (see top of this file for reference)
    N = int(segments * (segments - 1) / 2)
    rij = np.zeros((N, 2))
    A = np.zeros((N, segments - 1))
    flag = 0
    for i in range(segments - 1):
        for j in range(i + 1, segments):
            rij[flag, 0] = shifts[i, j, 1]
            rij[flag, 1] = shifts[i, j, 0]
            A[flag, i:j] = 1
            flag += 1
    Dj = np.dot(np.linalg.pinv(A), rij)
    shift_y = np.insert(np.cumsum(Dj[:, 0]), 0, 0)
    shift_x = np.insert(np.cumsum(Dj[:, 1]), 0, 0)

    # interpolate these timepoints to find drift for every frame.
    t = (starting_indices[1:] + starting_indices[:-1]) / 2
    all_shift_x = InterpolatedUnivariateSpline(t, shift_x, k=3)
    all_shift_y = InterpolatedUnivariateSpline(t, shift_y, k=3)
    t_inter = np.arange(n_frames + 1)

    drift = (all_shift_x(t_inter) * pixel_size, all_shift_y(t_inter) * pixel_size)
    return drift
