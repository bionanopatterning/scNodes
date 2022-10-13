from node import *


def create():
    return CropImageNode()


class CropImageNode(Node):
    title = "Crop image"
    group = "Image processing"
    colour = (143 / 255, 123 / 255, 103 / 255, 1.0)

    def __init__(self):
        super().__init__(Node.TYPE_CROP_IMAGE)
        self.size = [140, 100]
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)

        self.roi = [0, 0, 0, 0]
        self.use_roi = True

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            if self.frame_requested_by_image_viewer:
                return data_source.get_image(idx)
            else:
                out_frame = data_source.get_image(idx).clone()
                pxd = out_frame.load()[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                out_frame.data = pxd
                out_frame.width, out_frame.height = out_frame.data.shape
                return out_frame
