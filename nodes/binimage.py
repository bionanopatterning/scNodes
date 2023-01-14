from node import *


def create():
    return BinImageNode()


class BinImageNode(Node):
    title = "Bin image"
    group = "Image processing"
    colour = (143 / 255, 143 / 255, 143 / 255, 1.0)
    sortid = 110
    MODES = ["Average", "Median", "Min", "Max", "Sum"]

    def __init__(self):
        super().__init__()
        self.size = 170
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, self, [ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_DATASET])
        self.output = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.OUTPUT, self, [ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_DATASET])
        self.factor = 2
        self.mode = 0

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.output.render_start()
            self.dataset_in.render_end()
            self.output.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(100)
            _c, self.mode = imgui.combo("Method", self.mode, BinImageNode.MODES)
            self.any_change = _c or self.any_change
            imgui.pop_item_width()
            imgui.push_item_width(60)
            _c, self.factor = imgui.input_int("Bin factor", self.factor, 0, 0)
            self.any_change = _c or self.any_change
            self.factor = max([self.factor, 1])
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            image_in = data_source.get_image(idx)
            pxd = image_in.load()
            width, height = pxd.shape
            ## TODO: fix binning
            pxd = pxd[:self.factor * (width // self.factor), :self.factor * (height // self.factor)]
            if self.mode == 0:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).mean(2).mean(0)
            elif self.mode == 1:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).median(2).median(0)
            elif self.mode == 2:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).min(2).min(0)
            elif self.mode == 3:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).max(2).max(0)
            elif self.mode == 4:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).sum(2).sum(0)
            image_in.data = pxd
            return image_in

