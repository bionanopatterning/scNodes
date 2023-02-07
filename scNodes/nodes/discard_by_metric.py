from scNodes.core.node import *


def create():
    return DiscardFramesNode()


class DiscardFramesNode(Node):
    title = "Discard frames by metric"
    colour = (145/255, 236/255, 54/255, 1.0)
    group = "Image processing"
    sortid = 110
    enabled = False

    def __init__(self):
        super().__init__()

        self.connectable_attributes["dataset_in"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes["dataset_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)

    def render(self):
        if super().render_start():
            self.connectable_attributes["dataset_in"].render_start()
            self.connectable_attributes["dataset_out"].render_start()
            self.connectable_attributes["dataset_in"].render_end()
            self.connectable_attributes["dataset_out"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            super().render_end()

    def get_image_impl(self, idx=None):
        return None

