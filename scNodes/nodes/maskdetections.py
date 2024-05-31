from scNodes.core.node import *

def create():
    return MaskDetectionsNode()

class MaskDetectionsNode(Node):
    description = "Using a binary mask as the input, filter an incoming frame's particle list by removing those where mask != 1"
    title = "Mask detections"
    group = ["PSF-fitting reconstruction"]
    colour = (1.0, 0.0, 0.0, 1.0)
    enabled = False

    def __init__(self):
        super().__init__()
        self.size = 250

        self.connectable_attributes["coordinates_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["image_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["coordinates_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.OUTPUT, parent=self)

        self.infotext = "Masking particles."
        self.params["invert"] = False
        self.mask = None

    def render(self):
        if super().render_start():
            self.connectable_attributes["coordinates_in"].render_start()
            self.connectable_attributes["coordinates_out"].render_start()
            self.connectable_attributes["coordinates_in"].render_end()
            self.connectable_attributes["coordinates_out"].render_end()
            imgui.spacing()
            self.connectable_attributes["image_in"].render_start()
            self.connectable_attributes["image_in"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _, self.params["invert"] = imgui.checkbox("invert mask", self.params["invert"])
            self.any_change = self.any_change or _
            if self.any_change:
                self.set_mask()
            super().render_end()

    def set_mask(self):
        mask_source = self.connectable_attributes["image_in"].get_incoming_node()
        if mask_source:
            mask = mask_source.get_image(None).load()
            self.mask = (mask > 0).astype(np.float32)
            if self.params["invert"]:
                self.mask = 1 - self.mask

    def get_roi(self):
        data_source = self.connectable_attributes["coordinates_in"].get_incoming_node()
        if data_source:
            return data_source.get_roi()

    def get_image_impl(self, idx=None):
        data_source = self.connectable_attributes["coordinates_in"].get_incoming_node()
        if self.mask is None:
            self.set_mask()
        if data_source:
            input_frame = data_source.get_image(idx)
            particle_mask = np.zeros_like(self.mask)
            for c in input_frame.maxima:
                particle_mask[c[0], c[1]] = 1
            particle_mask = particle_mask * self.mask
            new_maxima = [(i, j) for i, j in np.transpose(np.where(particle_mask))]
            out_frame = input_frame.clone()
            out_frame.data = self.mask
            out_frame.maxima = np.array(new_maxima)
            return out_frame




