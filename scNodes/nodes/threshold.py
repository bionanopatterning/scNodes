from scNodes.core.node import *
from skimage import filters

def create():
    return ThresholdNode()


class ThresholdNode(Node):
    title = "Threshold"
    colour = (0.8, 0.5, 0.0, 1.0)
    group = "Image processing"
    #sortid = 110
    THRESH_METHODS = ["Fixed", "Auto Global", "Auto Frame"]
    AUTO_METHODS = ["percentage", "percentile", "Otsu"]

    def __init__(self):
        super().__init__()

        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)

        self.params["method"] = 0
        self.params["threshold"] = 0
        self.params["auto_method"] = 0
        self.params["percentage"] = 50
        self.params["percentile"] = 50

        self.autocontrast = True


    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_out"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            self.connectable_attributes["dataset_out"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.set_next_item_width(50)
            _c, self.params["threshold"] = imgui.input_int("Threshold Level", self.params["threshold"], 0, 0)
            if _c:
                self.autocontrast = False
            self.any_change = self.any_change or _c

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text("Method")
            imgui.push_item_width(170)
            _c, self.params["auto_method"] = imgui.combo("##auto_method", self.params["auto_method"], ThresholdNode.AUTO_METHODS)
            self.any_change = self.any_change or _c

            if self.params["auto_method"] == 0:
                imgui.push_item_width(85)
                _c, self.params["percentage"] = imgui.slider_float("percentage", self.params["percentage"], 0, 100, format="%.0f")
                self.any_change = self.any_change or _c
            elif self.params["auto_method"] == 1:
                imgui.push_item_width(85)
                _c, self.params["percentile"] = imgui.slider_float("percentile", self.params["percentile"], 0, 100, format="%.0f")
                self.any_change = self.any_change or _c

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _always_auto_changed, self.autocontrast = imgui.checkbox("always auto", self.autocontrast)
            self.any_change = self.any_change or _always_auto_changed

            if imgui.button("Auto once", width=80, height=19):
                self.autocontrast = False
                self._compute_auto_contrast()
                self.any_change = True

            if imgui.button("Auto Stack once", width=80, height=19):
                self.autocontrast = False
                self._compute_auto_contrast(None, stack=True)
                self.any_change = True

            super().render_end()

    def _compute_auto_contrast(self, idx=None, stack=False):
        incoming_node = self.connectable_attributes["dataset_in"].get_incoming_node()
        if incoming_node:

            if stack: # Global
                frame = incoming_node.get_image(0)
                n_frames = Node.get_source_load_data_node(self).dataset.n_frames
                data = np.zeros((frame.width, frame.height, n_frames))
                for i in range(n_frames):
                    data[:, :, i] = incoming_node.get_image(i).load()

            else:
                incoming_frame = incoming_node.get_image(idx)
                data = incoming_frame.load()

            if self.params["auto_method"] == 0:
                p = self.params["percentage"]
                self.params["threshold"] = int(((100 - p) * np.amin(data) + p * np.amax(data)) / 100)

            elif self.params["auto_method"] == 1:
                self.params["threshold"] = np.percentile(data, self.params["percentile"])

            elif self.params["auto_method"] == 2:
                self.params["threshold"] = filters.threshold_otsu(data)

    def get_image_impl(self, idx=None):
        incoming_node = self.connectable_attributes["dataset_in"].get_incoming_node()
        if incoming_node:
            incoming_frame = incoming_node.get_image(idx)
            data = incoming_frame.load()

            if self.autocontrast:
                self._compute_auto_contrast(idx)

            output_data = incoming_frame.clone()
            output_data.data = data > self.params["threshold"]

            return output_data
