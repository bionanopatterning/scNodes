from node import *


def create():
    return CustomNode()


class CustomNode(Node):
    title = "This is an example of a custom node"  # The title is the name by which the node appears in the programme.
    group = ["Image processing", "Custom"]  # In the 'add node' right-click context menu, nodes appear grouped. A node can be in multiple groups, as in this example. Any group name can be used here - also groups that are not in the default software.
    colour = (1.0, 0.0, 1.0, 1.0)  # this node will be magenta.

    def __init__(self):
        super().__init__(Node.TYPE_GET_IMAGE) ## TODO: get rid of node type integers.
        self.size = [200, 120] ## TODO: specify horizontal size only - vertical is done automatically.

        ## defining in- and output attributes.
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes.append(self.dataset_in) ## TODO: do self.connectable_attributes.append(...) in the ConnectableAttribute __init__!


    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.dataset_in.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text("This custom node doesn't do\nanything yet...")
            self.tooltip("but it does have a tooltip.")

            super().render_end()


    def on_update(self):
        pass


    def get_image_impl(self, idx=None):
        pass
