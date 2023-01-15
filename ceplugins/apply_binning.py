from ceplugin import *
from copy import deepcopy

def create():
    return BinPlugin()


class BinPlugin(CEPlugin):
    title = "Apply binning"
    description = "Create a binned copy of a frame."

    MODES = ["Average", "Median", "Min", "Max", "Sum"]
    
    def __init__(self):
        self.frame = None
        self.keep_original = False
        self.mode = 0
        self.factor = 2
        
    def render(self):
        imgui.text("Selected frame:")
        self.frame = self.widget_show_active_frame_title()
        imgui.push_item_width(100)
        _, self.mode = imgui.combo("Mode", self.mode, BinPlugin.MODES)
        _, self.factor = imgui.input_int("Factor", self.factor, 0, 0)
        imgui.pop_item_width()
        _, self.keep_original = imgui.checkbox("Keep original", self.keep_original)
        self.factor = max([self.factor, 1])
        if self.centred_button("Convert"):
            if self.frame is not None:
                binned_frame = BinPlugin.bin_clemframe(self.frame, self.factor, self.mode)
                binned_frame.update_lut()
                cfg.ce_frames.append(binned_frame)
                binned_frame.move_to_front()
                binned_frame.transform = deepcopy(self.frame.transform)
                if not self.keep_original:
                    cfg.correlation_editor.delete_frame(self.frame)

    @staticmethod
    def bin_clemframe(frame, factor, mode):
        if frame.is_rgb:
            binned = np.empty((frame.data.shape[0] // factor, frame.data.shape[1] // factor, 3))
            binned[:, :, 0] = BinPlugin.bin_grayscale(frame.data[:, :, 0], factor, mode)
            binned[:, :, 1] = BinPlugin.bin_grayscale(frame.data[:, :, 0], factor, mode)
            binned[:, :, 2] = BinPlugin.bin_grayscale(frame.data[:, :, 0], factor, mode)
        else:
            binned = BinPlugin.bin_grayscale(frame.data, factor, mode)
        outframe = CLEMFrame(binned)
        outframe.pixel_size = frame.pixel_size * factor
        outframe.lut = copy(frame.lut)
        outframe.contrast_lims = copy(frame.contrast_lims)
        outframe.lut_clamp_mode = copy(frame.lut_clamp_mode)
        outframe.pivot_point = copy(frame.pivot_point)
        outframe.title = frame.title + f"_bin{factor}"
        return outframe

    @staticmethod
    def bin_grayscale(array, factor, mode):
        width, height = array.shape
        pxd = array[:factor * (width // factor), :factor * (height // factor)]
        if mode == 0:
            pxd = pxd.reshape((width // factor, factor, height // factor, factor)).mean(3).mean(1)
        elif mode == 1:
            pxd = pxd.reshape((width // factor, factor, height // factor, factor))
            pxd = np.median(pxd, axis=(3, 1))
        elif mode == 2:
            pxd = pxd.reshape((width // factor, factor, height // factor, factor)).min(3).min(1)
        elif mode == 3:
            pxd = pxd.reshape((width // factor, factor, height // factor, factor)).max(3).max(1)
        elif mode == 4:
            pxd = pxd.reshape((width // factor, factor, height // factor, factor)).sum(3).sum(1)
        return pxd