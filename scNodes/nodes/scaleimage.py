from scNodes.core.node import *
import cv2


def create():
    return ScaleImageNode()


class ScaleImageNode(Node):
    description = "This node changes the size (in terms of pixel width / height) of the input frames."
    title = "Scale image"
    group = "Image processing"
    colour = (143 / 255, 143 / 255, 143 / 255, 1.0)
    sortid = 110
    CV2_INTERPOLATION_METHODS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]

    def __init__(self):
        super().__init__()
        self.size = 170
        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, self, [ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_DATASET])
        self.connectable_attributes["dataset_in"].colour = ConnectableAttribute.COLOUR[ConnectableAttribute.TYPE_DATASET]
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.OUTPUT, self, [ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_DATASET])
        self.connectable_attributes["dataset_out"].colour = ConnectableAttribute.COLOUR[ConnectableAttribute.TYPE_DATASET]
        self.params["fac_x"] = 0.60
        self.params["fac_y"] = 0.60
        self.params["interpolation"] = 0

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_out"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            self.connectable_attributes["dataset_out"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text("Scale factors:")
            imgui.push_item_width(40)
            _c, self.params["fac_x"] = imgui.input_float("x", self.params["fac_x"], format="%.2f")
            self.any_change = _c or self.any_change
            if _c:
                self.params["fac_y"] = self.params["fac_x"]
            _c, self.params["fac_y"] = imgui.input_float("y", self.params["fac_y"], format="%.2f")
            self.any_change = _c or self.any_change
            imgui.pop_item_width()
            imgui.spacing()
            imgui.text("Interpolation:")
            imgui.set_next_item_width(150)
            _c, self.params["interpolation"] = imgui.combo("##interp", self.params["interpolation"], ["Nearest neighbour", "Linear", "cv2 INTER_AREA", "Cubic"])
            self.any_change = _c or self.any_change
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.connectable_attributes["dataset_in"].get_incoming_node()
        if data_source:
            outframe = data_source.get_image(idx).clone()
            pxd = outframe.load()
            outsize = (int(np.shape(pxd)[0] * self.params["fac_x"]), int(np.shape(pxd)[1] * self.params["fac_y"]))
            outframe.data = cv2.resize(pxd, outsize, self.params["fac_x"], self.params["fac_y"], ScaleImageNode.CV2_INTERPOLATION_METHODS[self.params["interpolation"]])
            outframe.pixel_size /= self.params["fac_x"]
            outframe.width, outframe.height = np.shape(outframe.data)[0:2]
            return outframe

