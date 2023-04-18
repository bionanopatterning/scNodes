from scNodes.core.node import *
from tkinter import filedialog
import dill as pickle

def create():
    return LoadReconstructionNode()


class LoadReconstructionNode(Node):
    description = "Load a Reconstruction from a .csv file. Compatible with .csv files that are structured like\n" \
                  "those exported by ThunderSTORM (or scNodes.)"
    title = "Import reconstruction"
    group = "Data IO"
    colour = (213 / 255, 10 / 255, 70 / 255, 1.0)
    sortid = 4

    def __init__(self):
        super().__init__()
        self.size = 200

        self.connectable_attributes["reconstruction_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.OUTPUT, self)
        self.particle_data = None
        self.params["path"] = ""

    def render(self):
        if super().render_start():
            self.connectable_attributes["reconstruction_out"].render_start()
            self.connectable_attributes["reconstruction_out"].render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.text("Select source file")
            imgui.push_item_width(150)
            _, self.params["path"] = imgui.input_text("##intxt", self.params["path"], 256, imgui.INPUT_TEXT_ALWAYS_OVERWRITE)
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("...", 26, 19):
                selected_file = filedialog.askopenfilename(filetype=[("Reconstruction", ".csv"), ("Reconstruction", ".recon")])
                if selected_file is not None:
                    self.params["path"] = selected_file
                    self.on_select_file()
            super().render_end()

    def on_receive_drop(self, files):
        self.params["path"] = files[0]
        self.on_select_file()

    def on_select_file(self):
        try:
            self.particle_data = ParticleData.from_csv(self.params["path"])
        except Exception as e:
            cfg.set_error(e, "Error importing reconstruction. Reconstruction should be a .csv file.")

    def get_particle_data_impl(self):
        try:
            self.particle_data.clean()
            return self.particle_data
        except Exception as e:
            cfg.set_error(e, f"Error getting reconstruction from {self.title} node")
            return None
