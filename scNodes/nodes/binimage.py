from scNodes.core.node import *


def create():
    return BinImageNode()


class BinImageNode(Node):
    description = "Outputs binned copies of the input frames."
    title = "Bin image"
    group = "Image processing"
    colour = (143 / 255, 143 / 255, 143 / 255, 1.0)
    sortid = 110
    MODES = ["Average", "Median", "Min", "Max", "Sum"]

    def __init__(self):
        super().__init__()
        
        self.size = 170
        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, self, [ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_DATASET])
        self.params["output"] = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.OUTPUT, self, [ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_DATASET])
        self.params["factor"] = 2
        self.params["mode"] = 0

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.params["output"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            self.params["output"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(100)
            _c, self.params["mode"] = imgui.combo("Method", self.params["mode"], BinImageNode.MODES)
            self.any_change = _c or self.any_change
            imgui.pop_item_width()
            imgui.push_item_width(60)
            _c, self.params["factor"] = imgui.input_int("Bin factor", self.params["factor"], 0, 0)
            self.any_change = _c or self.any_change
            self.params["factor"] = max([self.params["factor"], 1])
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.connectable_attributes["dataset_in"].get_incoming_node()
        if data_source:
            image_in = data_source.get_image(idx)
            pxd = image_in.load()
            width, height = pxd.shape
            pxd = pxd[:self.params["factor"] * (width // self.params["factor"]), :self.params["factor"] * (height // self.params["factor"])]
            if self.params["mode"] == 0:
                pxd = pxd.reshape((width // self.params["factor"], self.params["factor"], height // self.params["factor"], self.params["factor"])).mean(3).mean(1)
            elif self.params["mode"] == 1:
                pxd = pxd.reshape((width // self.params["factor"], self.params["factor"], height // self.params["factor"], self.params["factor"]))
                pxd = np.median(pxd, axis=(3, 1))
            elif self.params["mode"] == 2:
                pxd = pxd.reshape((width // self.params["factor"], self.params["factor"], height // self.params["factor"], self.params["factor"])).min(3).min(1)
            elif self.params["mode"] == 3:
                pxd = pxd.reshape((width // self.params["factor"], self.params["factor"], height // self.params["factor"], self.params["factor"])).max(3).max(1)
            elif self.params["mode"] == 4:
                pxd = pxd.reshape((width // self.params["factor"], self.params["factor"], height // self.params["factor"], self.params["factor"])).sum(3).sum(1)
            image_out = image_in.clone()
            image_out.data = pxd
            return image_out

