from scNodes.core.node import *
from tkinter import filedialog
from scNodes.core.util import get_filetype


def create():
    return LoadImageNode()


class LoadImageNode(Node):
    description = "Load a single .tif image."
    title = "Import image"
    group = "Data IO"
    colour = (84 / 255, 77 / 255, 222 / 255, 1.0)
    sortid = 3

    def __init__(self):
        super().__init__()
        self.size = 200

        # Set up connectable attributes
        self.connectable_attributes["image_out"] = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.OUTPUT, parent=self)

        self.params["path"] = ""
        self.img = Frame("virtual_frame")
        self.img.width = 1
        self.img.height = 1

        self.NODE_RETURNS_IMAGE = False

    def render(self):
        if super().render_start():
            self.connectable_attributes["image_out"].render_start()
            self.connectable_attributes["image_out"].render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.text("Select source file")
            imgui.push_item_width(150)
            _enter, self.params["path"] = imgui.input_text("##intxt", self.params["path"], 256, imgui.INPUT_TEXT_ALWAYS_OVERWRITE | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("...", 26, 19):
                selected_file = filedialog.askopenfilename()
                if selected_file is not None:
                    if get_filetype(selected_file) in ['.tiff', '.tif']:
                        self.params["path"] = selected_file
                        self.on_select_file()
            elif _enter:
                self.on_select_file()
            imgui.text(f"image size: {self.img.width}x{self.img.height}")

            super().render_end()

    def on_select_file(self):
        try:
            self.img = Frame(self.params["path"])
            self.any_change = True
            self.NODE_RETURNS_IMAGE = True
        except Exception as e:
            cfg.set_error(e, f'Error importing "{self.params["path"]}"')

    def get_image_impl(self, idx):
        return self.img
    
    def on_load(self):
        self.on_select_file()
