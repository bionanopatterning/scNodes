from node import *
import cv2


def create():
    return ScaleImageNode()


class ScaleImageNode(Node):
    title = "Scale image"
    group = "Image processing"
    colour = (143 / 255, 143 / 255, 143 / 255, 1.0)
    sortid = 110
    CV2_INTERPOLATION_METHODS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]

    def __init__(self):
        super().__init__()
        self.size = 170
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, self, [ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_DATASET])
        self.dataset_in.colour = ConnectableAttribute.COLOUR[ConnectableAttribute.TYPE_DATASET]
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.OUTPUT, self, [ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_DATASET])
        self.dataset_out.colour = ConnectableAttribute.COLOUR[ConnectableAttribute.TYPE_DATASET]
        self.fac_x = 0.60
        self.fac_y = 0.60
        self.interpolation = 0

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.dataset_out.render_start()
            self.dataset_in.render_end()
            self.dataset_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text("Scale factors:")
            imgui.push_item_width(40)
            _c, self.fac_x = imgui.input_float("x", self.fac_x, format="%.2f")
            self.any_change = _c or self.any_change
            if _c:
                self.fac_y = self.fac_x
            _c, self.fac_y = imgui.input_float("y", self.fac_y, format="%.2f")
            self.any_change = _c or self.any_change
            imgui.pop_item_width()
            imgui.spacing()
            imgui.text("Interpolation:")
            imgui.set_next_item_width(150)
            _c, self.interpolation = imgui.combo("##interp", self.interpolation, ["Nearest neighbour", "Linear", "cv2 INTER_AREA", "Cubic"])
            self.any_change = _c or self.any_change
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            outframe = data_source.get_image(idx).clone()
            pxd = outframe.load()
            outsize = (int(np.shape(pxd)[0] * self.fac_x), int(np.shape(pxd)[1] * self.fac_y))
            outframe.data = cv2.resize(pxd, outsize, self.fac_x, self.fac_y, ScaleImageNode.CV2_INTERPOLATION_METHODS[self.interpolation])
            outframe.pixel_size /= self.fac_x
            outframe.width, outframe.height = np.shape(outframe.data)[0:2]
            return outframe

