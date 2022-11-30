from node import *
from skimage.feature import peak_local_max

def create():
    return LoadReconstructionNode()


class LoadReconstructionNode(Node):
    title = "Import reconstruction"
    group = "Data IO"
    colour = (213 / 255, 10 / 255, 70 / 255, 1.0)
    sortid = 4

    def __init__(self):
        super().__init__()
        self.size = 200

        self.reconstruction_out = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.OUTPUT, self)


    def render(self):
        if super().render_start():
            self.reconstruction_out.render_start()
            self.reconstruction_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text("todo ...") # TODO
            super().render_end()
