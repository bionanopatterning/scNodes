from ceplugin import *
from copy import deepcopy

def create():
    return SplitRGBPlugin()


class SplitRGBPlugin(CEPlugin):
    title = "Split RGB"
    description = "Split an RGB frame into the three underlying grayscale images."

    def __init__(self):
        self.rgb_frame = None
        self.keep_original = False

    def render(self):
        _c, self.rgb_frame = self.widget_select_frame_rgb("RGB Frame:", self.rgb_frame)
        _c, self.keep_original = imgui.checkbox("Keep original", self.keep_original)
        if self.centred_button("Convert"):
            rgb = self.rgb_frame.data
            r = CLEMFrame(rgb[:, :, 0])
            r.pixel_size = self.rgb_frame.pixel_size
            r.transform = deepcopy(self.rgb_frame.transform)
            r.lut = 0
            r.colour = (1.0, 0.0, 0.0, 1.0)
            r.update_lut()
            g = CLEMFrame(rgb[:, :, 1])
            g.pixel_size = self.rgb_frame.pixel_size
            g.transform = deepcopy(self.rgb_frame.transform)
            g.lut = 0
            g.colour = (0.0, 1.0, 0.0, 1.0)
            g.update_lut()
            b = CLEMFrame(rgb[:, :, 2])
            b.pixel_size = self.rgb_frame.pixel_size
            b.transform = deepcopy(self.rgb_frame.transform)
            b.lut = 0
            b.colour = (0.0, 0.0, 1.0, 1.0)
            b.update_lut()
            cfg.ce_frames.append(r)
            cfg.ce_frames.append(g)
            cfg.ce_frames.append(b)
            b.move_to_front()
            g.move_to_front()
            r.move_to_front()
            if not self.keep_original:
                cfg.correlation_editor.delete_frame(self.rgb_frame)


