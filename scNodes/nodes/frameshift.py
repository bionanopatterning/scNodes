from scNodes.core.node import *


def create():
    return FrameShiftNode()


class FrameShiftNode(Node):
    title = "Frame shift"
    group = "Image processing"
    colour = (235 / 255, 232 / 255, 80 / 255, 1.0)
    sortid = 104

    def __init__(self):
        super().__init__()
        self.size = 140

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.shift = 0

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(80)
            _c, self.shift = imgui.input_int("shift", self.shift, 1, 10)
            self.any_change = _c or self.any_change

            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            return data_source.get_image(idx + self.shift)
