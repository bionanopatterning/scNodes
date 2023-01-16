import imgui.internal

from src.ceplugin import *
from scipy.ndimage import gaussian_filter

def create():
    return BlurPlugin()


class BlurPlugin(CEPlugin):
    title = "Blur frames"
    description = "Apply a Gaussian filter to a frame. This plugin changes the pixel data of a frame object.\n" \
                  "Be sure to duplicate the frame first if you want to keep the original frame in the scene"

    def __init__(self):
        self.selected_frame = None
        self.sigma = 1.0  # blur kernel standard deviation in pixels.

    def render(self):
        self.selected_frame = CEPlugin.widget_show_active_frame_title(label="Selected frame:")

        imgui.set_next_item_width(50)
        _, self.sigma = imgui.input_float("Sigma", self.sigma)

        if CEPlugin.widget_centred_button("Blur"):
            self.process_frame()

    def process_frame(self):
        out_frame = self.selected_frame.duplicate()
        img_data = out_frame.data

        if self.selected_frame.is_rgb:
            img_data[:, :, 0] = gaussian_filter(img_data[:, :, 0], self.sigma)
            img_data[:, :, 1] = gaussian_filter(img_data[:, :, 1], self.sigma)
            img_data[:, :, 2] = gaussian_filter(img_data[:, :, 2], self.sigma)
        else:
            img_data[:, :] = gaussian_filter(img_data, self.sigma)
        out_frame.data = img_data

        cfg.ce_frames.append(out_frame)
        cfg.ce_frames.remove(self.selected_frame)
        out_frame.update()