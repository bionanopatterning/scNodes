from node import *
from tkinter import filedialog


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
        self.particle_data = None
        self.path = ""

    def render(self):
        if super().render_start():
            self.reconstruction_out.render_start()
            self.reconstruction_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text("Select source file")
            imgui.push_item_width(150)
            _, self.path = imgui.input_text("##intxt", self.path, 256, imgui.INPUT_TEXT_ALWAYS_OVERWRITE)
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("...", 26, 19):
                selected_file = filedialog.askopenfilename(filetype=[("Reconstruction", ".csv")])
                if selected_file is not None:
                    self.path = selected_file
                    self.on_select_file()
            super().render_end()

    def on_select_file(self):
        print("smik")
        try:
            self.particle_data = ParticleData.from_csv(self.path)
        except Exception as e:
            cfg.set_error(e, "Error importing reconstruction. Reconstruction should be a .csv file.")

    def get_particle_data_impl(self):
        self.particle_data.clean()
        return self.particle_data
