from src.node import *


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
            pxd = pxd[:self.factor * (width // self.factor), :self.factor * (height // self.factor)]
            if self.mode == 0:
                pxd = pxd.reshape((width // self.factor, self.factor, height // self.factor, self.factor)).mean(3).mean(1)
            elif self.mode == 1:
                pxd = pxd.reshape((width // self.factor, self.factor, height // self.factor, self.factor))
                pxd = np.median(pxd, axis=(3, 1))
            elif self.mode == 2:
                pxd = pxd.reshape((width // self.factor, self.factor, height // self.factor, self.factor)).min(3).min(1)
            elif self.mode == 3:
                pxd = pxd.reshape((width // self.factor, self.factor, height // self.factor, self.factor)).max(3).max(1)
            elif self.mode == 4:
                pxd = pxd.reshape((width // self.factor, self.factor, height // self.factor, self.factor)).sum(3).sum(1)
            return image_in

