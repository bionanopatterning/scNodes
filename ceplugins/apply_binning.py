from ceplugin import *
from copy import deepcopy

def create():
    return BinPlugin()


class BinPlugin(CEPlugin):
    title = "Apply binning"
    description = "Apply binning to an image in memory, rather than in rendering (which is the default in the Correlation Editor)"

    MODES = ["Average", "Median", "Min", "Max", "Sum"]
    
    def __init__(self):
        self.frame = None
        self.keep_original = False
        self.mode = 0
        self.factor = 2
        
    def render(self):
        _, self.frame = self.widget_select_frame_any("Frame:", self.frame)
        _, self.keep_original = imgui.checkbox("Keep original", self.keep_original)
        imgui.push_item_width(100)
        _, self.mode = imgui.combo("Mode", self.mode, BinPlugin.MODES)
        _, self.factor = imgui.input_int("Factor", self.factor, 0, 0)
        imgui.pop_item_width()
        self.factor = max([self.factor, 1])
        if self.centred_button("Convert"):
            binned_frame = BinPlugin.bin_clemframe(self.frame, self.factor, self.mode)
            binned_frame.update_lut()
            cfg.ce_frames.append(binned_frame)
            binned_frame.move_to_front()
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
        outframe.title = frame.title + f"_bin{factor}"
        return outframe

    @staticmethod
    def bin_grayscale(array, factor, mode):
        width, height = array.shape
        pxd = array[:factor * (width // factor), :factor * (height // factor)]
        if mode == 0:
            pxd = pxd.reshape((factor, width // factor, factor, height // factor)).mean(2).mean(0)
        elif mode == 1:
            pxd = pxd.reshape((factor, width // factor, factor, height // factor)).median(2).median(0)
        elif mode == 2:
            pxd = pxd.reshape((factor, width // factor, factor, height // factor)).min(2).min(0)
        elif mode == 3:
            pxd = pxd.reshape((factor, width // factor, factor, height // factor)).max(2).max(0)
        elif mode == 4:
            pxd = pxd.reshape((factor, width // factor, factor, height // factor)).sum(2).sum(0)
        return pxd