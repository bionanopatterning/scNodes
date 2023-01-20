from scNodes.core.node import *


def create():
    return ImageCalculatorNode()


class ImageCalculatorNode(Node):
    title = "Image calculator"
    group = "Image processing"
    colour = (143 / 255, 143 / 255, 143 / 255, 1.0)
    sortid = 103
    # Note: the output dataset has all the metadata of dataset_in_a
    OPERATIONS = ["Add", "Subtract", "Divide", "Multiply"]

    def __init__(self):
        super().__init__()
        self.size = 230
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, parent=self, allowed_partner_types=[ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.TYPE_IMAGE])
        self.input_b = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, parent=self, allowed_partner_types=[ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.TYPE_IMAGE])
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)
        self.image_out = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.OUTPUT, parent=self)

        self.operation = 1

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            if self.dataset_in.current_type == ConnectableAttribute.TYPE_IMAGE:
                self.image_out.render_start()
                self.image_out.render_end()
            else:
                self.dataset_out.render_start()
                self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            self.input_b.render_start()
            self.input_b.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.push_item_width(90)
            _c, self.operation = imgui.combo("Operation", self.operation, ImageCalculatorNode.OPERATIONS)
            self.any_change = self.any_change | _c
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        try:
            source_a = self.dataset_in.get_incoming_node()
            source_b = self.input_b.get_incoming_node()
            if source_a and source_b:
                img_a = source_a.get_image(idx)
                img_b = source_b.get_image(idx)
                img_a_pxd = img_a.load()
                img_b_pxd = img_b.load()

                w = min([img_a_pxd.shape[0], img_b_pxd.shape[0]])
                h = min([img_a_pxd.shape[1], img_b_pxd.shape[1]])

                img_a_pxd = img_a_pxd[:w, :h]
                img_b_pxd = img_b_pxd[:w, :h]

                img_out = None
                if self.operation == 0:
                    img_out = img_a_pxd + img_b_pxd
                elif self.operation == 1:
                    img_out = img_a_pxd - img_b_pxd
                elif self.operation == 2:
                    img_out = img_a_pxd / img_b_pxd
                elif self.operation == 3:
                    img_out = img_a_pxd * img_b_pxd
                if self.dataset_in.current_type != ConnectableAttribute.TYPE_IMAGE:
                    img_a.data = img_out
                    return img_a
                else:
                    virtual_frame = Frame("virtual_frame")
                    virtual_frame.data = img_out.astype(np.uint16)
                    return virtual_frame
        except Exception as e:
            cfg.set_error(Exception(), "ImageCalculatorNode error:\n"+str(e))
