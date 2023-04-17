from scNodes.core.node import *


def create():
    return FrameShiftNode()


class FrameShiftNode(Node):
    description = "This node shifts the reading index of the input dataset. This means that when, e.g., frame '10'\n" \
                  "is requested (by the image viewer or a downstream node), this node returns, e.g., frame '11' in-\n" \
                  "stead. (This is an example of a frameshift of +1; any shift value is allowed).\n" \
                  "\n" \
                  "When requesting frames outside of the range of the dataset (e.g. at index -1), the index is clamped\n" \
                  "to be within that range; e.g., negative indices return frame 0, indices larger than the amount of\n" \
                  "frames in the dataset return the last frame of the dataset."
    title = "Frame shift"
    group = "Image processing"
    colour = (235 / 255, 232 / 255, 80 / 255, 1.0)
    sortid = 104

    def __init__(self):
        super().__init__()
        self.size = 140

        # Set up connectable attributes
        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.params["shift"] = 0

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_out"].render_start()
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_out"].render_end()
            self.connectable_attributes["dataset_in"].render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(80)
            _c, self.params["shift"] = imgui.input_int("shift", self.params["shift"], 1, 10)
            self.any_change = _c or self.any_change

            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.connectable_attributes["dataset_in"].get_incoming_node()
        if data_source:
            return data_source.get_image(idx + self.params["shift"])
