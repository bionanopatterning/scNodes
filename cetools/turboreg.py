import imgui
import config as cfg
from clemframe import CLEMFrame

def create():
    return TurboregTool()


class TurboregTool:
    title = "Register TEM"
    description = "Register images of equal or different sizes based on image intensity.\n" \
                  "Useful to register high-magnification TEM images to low-magnification\n" \
                  "images of the same region (e.g. to register Exposure/Search frames onto\n" \
                  "Overview images.)\n" \
                  "Requires manual input of approximate\n" \
                  "location of child within parent image."

    def __init__(self):
        self.template_frame_idx = 0
        self.child_frame_idx = 0
        self.world_x = 0.0
        self.world_y = 0.0

    def render(self):

        frame_list = list()
        for frame in cfg.ce_frames:
            frame_list.append(frame.title)

        _cw = imgui.get_content_region_available_width()
        imgui.push_item_width(_cw)
        imgui.text("Select template frame:")
        _c, self.template_frame_idx = imgui.combo("##Template frame", self.template_frame_idx, frame_list)
        imgui.text("Select child frame:")
        _c, self.child_frame_idx = imgui.combo("##Child frame", self.child_frame_idx, frame_list)
        imgui.pop_item_width()

        imgui.text("Approximate ROI:")
        imgui.text(f"x = {self.world_x:.1f}, y = {self.world_y:.1f}")
        imgui.same_line()
        imgui.button("o", 19, 19)
        #imgui.tooltip("Indicate the approximate position on the template image of the child image location. Click here,\n"
        #              "then click in the scene at the target.")


    # TODO: how to handle input for dynamically imported ceTools()?
