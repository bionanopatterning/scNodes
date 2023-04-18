from scNodes.core.node import *


def create():
    return InvertNode()


class InvertNode(Node):
    description = "This is the node that is created in the 'Creating a custom node' tutorial that can be found in the\n" \
                  "user manual.\n" \
                  "\n" \
                  "The node inverts the pixel intensities of the input frames."
    title = "Invert images"
    colour = (0.8, 0.5, 0.0, 1.0)
    group = "Image processing"
    sortid = 110

    def __init__(self):
        super().__init__()

        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)

        self.use_roi = False

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_out"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            self.connectable_attributes["dataset_out"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _changed, self.use_roi = imgui.checkbox("use ROI", self.use_roi)
            self.any_change = self.any_change or _changed

            super().render_end()

    def get_image_impl(self, idx=None):
        incoming_node = self.connectable_attributes["dataset_in"].get_incoming_node()
        if incoming_node:
            incoming_frame = incoming_node.get_image(idx)
            data = incoming_frame.load()
            _roi = self.roi
            if not self.use_roi:
                _roi = [0, 0, incoming_frame.width, incoming_frame.height]
            roi_data = incoming_frame.load_roi(_roi)
            min_val = np.amin(roi_data)
            max_val = np.amax(roi_data)

            inverted_data = max_val + min_val - roi_data
            output_frame = incoming_frame.clone()
            output_frame.data = data
            output_frame.write_roi(_roi, inverted_data)
            return output_frame

