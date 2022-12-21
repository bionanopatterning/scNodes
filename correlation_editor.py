import imgui
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
from opengl_classes import *

class CorrelationEditor:
    if True:
        COLOUR_WINDOW_BACKGROUND = (0.94, 0.94, 0.94, 1.0)
        COLOUR_PANEL_BACKGROUND = (0.96, 0.96, 0.96, 0.96)
        COLOUR_TITLE_BACKGROUND = (0.87, 0.87, 0.83, 1.0)
        COLOUR_TEXT = (0.0, 0.0, 0.0, 1.0)

        WINDOW_ROUNDING = 5.0


    BLEND_MODES = ["Overlay", "Add", "Subtract"] # TODO: figure out which GL blend modes are useful

    def __init__(self, window, shared_font_atlas=None):
        # Note that CorrelationEditor and NodeEditor share a window. In either's on_update, end_frame,
        # __init__, etc. methods, bits of the same window-management related code are called.

        self.window = window
        self.window.clear_color = CorrelationEditor.COLOUR_WINDOW_BACKGROUND
        self.window.make_current()

        if shared_font_atlas is not None:
            self.imgui_context = imgui.create_context(shared_font_atlas)
        else:
            self.imgui_context = imgui.create_context()
        self.imgui_implementation = GlfwRenderer(self.window.glfw_window)
        self.window.set_mouse_callbacks()
        self.window.set_window_callbacks()

    def on_update(self):
        imgui.set_current_context(self.imgui_context)
        self.window.make_current()
        self.window.set_full_viewport()
        if self.window.focused:
            self.imgui_implementation.process_inputs()
        self.window.on_update()

        imgui.new_frame()

        ## content
        self.gui_main()
        ## end content

        imgui.render()
        self.imgui_implementation.render(imgui.get_draw_data())

    def gui_main(self):
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *CorrelationEditor.COLOUR_PANEL_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_TEXT, *CorrelationEditor.COLOUR_TEXT)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *CorrelationEditor.COLOUR_TITLE_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *CorrelationEditor.COLOUR_TITLE_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *CorrelationEditor.COLOUR_TITLE_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *CorrelationEditor.COLOUR_TITLE_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *CorrelationEditor.COLOUR_TITLE_BACKGROUND)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, CorrelationEditor.WINDOW_ROUNDING)

        self.frame_info_window()
        self.objects_info_window()

        imgui.pop_style_color(7)
        imgui.pop_style_var(1)

    def frame_info_window(self):
        imgui.begin("Image information", False, imgui.WINDOW_NO_RESIZE)

        imgui.end()

    def objects_info_window(self):
        pass

    def end_frame(self):
        self.window.end_frame()


class Renderer:
    pass


class CLEMFrame:
    def __init__(self, img_array):
        """Grayscale images only - img_array must be a 2D np.ndarray"""
        self.data = img_array
        self.width, self.height = self.data.shape
        self.texture = Texture(format="r32f")
        self.blend_mode = CorrelationEditor.BLEND_MODES[0]
        self.quad_va = VertexArray()


        self.pixel_size = 100.0  # pixel size in nm

        self.pivot_point = np.zeros(2)  # pivot point for rotation and scaling of this particular image. can be moved by the user.
        self.translation = np.zeros(2)
        self.rotation = 0.0
        self.scale = 1.0

    def get_model_matrix(self):
        scale_mat = self.scale * np.identity(3)
        scale_mat[2, 2] = 1.0

        rotation_mat = np.identity(3)
        _cos = np.cos(self.rotation * 180.0 / np.pi)
        _sin = np.sin(self.rotation * 180.0 / np.pi)
        rotation_mat[0, 0] = _cos
        rotation_mat[1, 0] = _sin
        rotation_mat[0, 1] = -_sin
        rotation_mat[1, 1] = _cos

        translation_mat = np.identity(3)
        translation_mat[0, 2] = self.translation[0]
        translation_mat[1, 2] = self.translation[1]

        return translation_mat * (rotation_mat * scale_mat)


