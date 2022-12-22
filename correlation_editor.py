import imgui
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
from opengl_classes import *
import settings
import time

class CorrelationEditor:
    if True:
        COLOUR_WINDOW_BACKGROUND = (0.94, 0.94, 0.94, 1.0)
        COLOUR_PANEL_BACKGROUND = (0.96, 0.96, 0.96, 0.96)
        COLOUR_TITLE_BACKGROUND = (0.87, 0.87, 0.83, 1.0)
        COLOUR_FRAME_BACKGROUND = (0.87, 0.87, 0.83, 1.0)
        COLOUR_FRAME_ACTIVE = (0.91, 0.91, 0.86, 1.0)
        COLOUR_FRAME_DARK = (0.83, 0.83, 0.76, 1.0)
        COLOUR_FRAME_EXTRA_DARK = (0.76, 0.76, 0.71, 1.0)

        COLOUR_TEXT = (0.0, 0.0, 0.0, 1.0)

        WINDOW_ROUNDING = 5.0


    BLEND_MODES = ["Overlay", "Add", "Subtract"] # TODO: figure out which GL blend modes are useful

    TRANSLATION_INCREMENT = 0.0
    TRANSLATION_INCREMENT_LARGE = 0.0
    ROTATION_INCREMENT = 0.0
    ROTATION_INCREMENT_LARGE = 0.0
    PIXEL_SIZE_INCREMENT = 0.0
    PIXEL_SIZE_INCREMENT_LARGE = 0.0

    WORLD_PIXEL_SIZE = 100.0
    HISTOGRAM_BINS = 50

    INFO_PANEL_WIDTH = 200
    INFO_HISTOGRAM_HEIGHT = 70
    INFO_LUT_PREVIEW_HEIGHT = 10

    TOOLTIP_APPEAR_DELAY = 1.0
    TOOLTIP_HOVERED_TIMER = 0.0
    TOOLTIP_HOVERED_START_TIME = 0.0

    #

    frames = list()

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

        ## DEBUG
        self.frames.append(CLEMFrame(np.random.randint(0, 255, (1024, 1024))))
        self.active_frame = self.frames[0]

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
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *CorrelationEditor.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *CorrelationEditor.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *CorrelationEditor.COLOUR_FRAME_ACTIVE)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *CorrelationEditor.COLOUR_FRAME_ACTIVE)
        imgui.push_style_color(imgui.COLOR_BUTTON, *CorrelationEditor.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *CorrelationEditor.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *CorrelationEditor.COLOUR_FRAME_EXTRA_DARK)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *CorrelationEditor.COLOUR_FRAME_ACTIVE)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *CorrelationEditor.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB, *CorrelationEditor.COLOUR_FRAME_DARK)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB_HOVERED, *CorrelationEditor.COLOUR_FRAME_DARK)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB_ACTIVE, *CorrelationEditor.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_BACKGROUND, *CorrelationEditor.COLOUR_FRAME_EXTRA_DARK)
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *CorrelationEditor.COLOUR_TEXT)
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, *CorrelationEditor.COLOUR_TEXT)
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *CorrelationEditor.COLOUR_FRAME_EXTRA_DARK)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, CorrelationEditor.WINDOW_ROUNDING)

        self.frame_info_window()
        self.objects_info_window()
        self.visuals_window()
        imgui.pop_style_color(23)
        imgui.pop_style_var(1)

    def frame_info_window(self):
        af = self.active_frame

        imgui.push_style_color(imgui.COLOR_HEADER, *CorrelationEditor.COLOUR_PANEL_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *CorrelationEditor.COLOUR_PANEL_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *CorrelationEditor.COLOUR_PANEL_BACKGROUND)
        imgui.set_next_window_size_constraints((CorrelationEditor.INFO_PANEL_WIDTH, -1), (CorrelationEditor.INFO_PANEL_WIDTH, -1))
        imgui.begin("Image information", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        #  Transform info
        expanded, _ = imgui.collapsing_header("Transform", None)
        if expanded:
            imgui.push_item_width(50)
            _, af.translation[0] = imgui.input_float("x", af.translation[0], CorrelationEditor.TRANSLATION_INCREMENT, CorrelationEditor.TRANSLATION_INCREMENT_LARGE, '%.1f nm', imgui.INPUT_TEXT_AUTO_SELECT_ALL)
            imgui.same_line(spacing=20)
            _, af.translation[1] = imgui.input_float("y", af.translation[0], CorrelationEditor.TRANSLATION_INCREMENT, CorrelationEditor.TRANSLATION_INCREMENT_LARGE, '%.1f nm', imgui.INPUT_TEXT_AUTO_SELECT_ALL)
            _, af.rotation = imgui.input_float("rotation", af.rotation, CorrelationEditor.ROTATION_INCREMENT, CorrelationEditor.ROTATION_INCREMENT_LARGE, '%.2f°', imgui.INPUT_TEXT_AUTO_SELECT_ALL)
            imgui.same_line(spacing=20)
            _, af.flip = imgui.checkbox("flip", af.flip)
            _, af.pixel_size = imgui.input_float("pixel size (nm)", af.pixel_size, CorrelationEditor.PIXEL_SIZE_INCREMENT, CorrelationEditor.PIXEL_SIZE_INCREMENT_LARGE, '%.1f', imgui.INPUT_TEXT_AUTO_SELECT_ALL)
            imgui.pop_item_width()
            imgui.separator()

        # Visuals info
        expanded, _ = imgui.collapsing_header("Visuals", None)
        if expanded:
            # LUT
            imgui.text("Look-up table")
            imgui.set_next_item_width(129)
            _clut, af.lut = imgui.combo("##LUT", af.lut, ["Custom colour"] + settings.lut_names, len(settings.lut_names) + 1)
            if af.lut == 0:
                imgui.same_line()
                _c, af.colour = imgui.color_edit4("##lutclr", *af.colour, imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL)
                _clut = _clut or _c
            if _clut:
                af.update_lut()

            imgui.same_line()
            if imgui.button("A", 20, 20):
                af.compute_autocontrast()
            CorrelationEditor.tooltip("Click to autocompute contrast limits.")

            # HISTOGRAM
            _cw = imgui.get_content_region_available_width()
            imgui.plot_histogram("##hist", af.hist_vals, graph_size=(_cw, CorrelationEditor.INFO_HISTOGRAM_HEIGHT))
            _l = af.hist_bins[0]
            _h = af.hist_bins[-1]
            _max = af.contrast_lims[1]
            _min = af.contrast_lims[0]
            _uv_left = 1.0 + (_l - _max) / (_max - _min)
            _uv_right = 1.0 + (_h - _max) / (_max - _min)
            imgui.image(af.lut_texture.renderer_id, _cw, CorrelationEditor.INFO_LUT_PREVIEW_HEIGHT, (_uv_left, 0.5), (_uv_right, 0.5), border_color=CorrelationEditor.COLOUR_FRAME_BACKGROUND)
            imgui.push_item_width(_cw)
            _c, af.contrast_lims[0] = imgui.slider_float("min", af.contrast_lims[0], af.hist_bins[0], af.hist_bins[-1], format='min %.1f')
            _c, af.contrast_lims[1] = imgui.slider_float("max", af.contrast_lims[1], af.hist_bins[0], af.hist_bins[-1], format='max %.1f')

            imgui.separator()

        imgui.end()
        imgui.pop_style_color(3)

    @staticmethod
    def tooltip(text):
        if imgui.is_item_hovered():
            if CorrelationEditor.TOOLTIP_HOVERED_TIMER == 0.0:
                CorrelationEditor.TOOLTIP_HOVERED_START_TIME = time.time()
                CorrelationEditor.TOOLTIP_HOVERED_TIMER = 0.001  # add a fake 1 ms to get out of this if clause
            elif CorrelationEditor.TOOLTIP_HOVERED_TIMER > CorrelationEditor.TOOLTIP_APPEAR_DELAY:
                imgui.set_tooltip(text)
            else:
                CorrelationEditor.TOOLTIP_HOVERED_TIMER = time.time() - CorrelationEditor.TOOLTIP_HOVERED_START_TIME
        if not imgui.is_any_item_hovered():
            CorrelationEditor.TOOLTIP_HOVERED_TIMER = 0.0

    def objects_info_window(self):
        pass

    def visuals_window(self):
        pass

    def end_frame(self):
        self.window.end_frame()


class Renderer:

    def __init__(self):
        self.quad_shader = Shader("shaders/ce_quad_shader.glsl")
        self.border_shader = Shader("shaders/ce_border_shader.glsl")

    def render_frame(self, frame):
        pass
        # TODO: figure out how to implement z-ordering. 1) Give quads a z coordinate, or 2) disable depth testing and manually manage rendering order
        # TODO: figure out whether to render to screen directly, or to FBO which is rendered onto a quad on the screen.
        # switch FBOs

        # render image

        # if frame is the active one, render border.

class Camera:

    def __init__(self):
        pass

    def get_vp_matrix(self):
        pass

class CLEMFrame:
    def __init__(self, img_array):
        """Grayscale images only - img_array must be a 2D np.ndarray"""
        self.data = img_array
        self.width, self.height = self.data.shape
        self.blend_mode = CorrelationEditor.BLEND_MODES[0]
        self.quad_va = VertexArray()
        self.border_va = VertexArray(attribute_format="xy")
        self.pixel_size = 100.0  # pixel size in nm
        self.pivot_point = np.zeros(2)  # pivot point for rotation and scaling of this particular image. can be moved by the user.
        self.translation = np.zeros(2)
        self.rotation = 0.0
        self.scale = 1.0  # automatically updated in get_model_matrix. Internal var to keep track of relative scale of quad.
        self.flip = False  # 1.0 or -1.0, in case the image is mirrored.

        # visuals
        self.lut = 1
        self.colour = (1.0, 1.0, 1.0, 1.0)
        self.opacity = 1.0
        self.contrast_lims = [0.0, 65535.0]
        self.compute_autocontrast()
        self.hist_bins = list()
        self.hist_vals = list()
        self.compute_histogram()

        # opengl
        self.texture = Texture(format="r32f")
        self.texture.update(self.data.astype(np.float32))
        self.lut_texture = Texture(format="rgb32f")
        self.update_lut()
        self.generate_va()

        # flags
        self.REQUIRES_UPDATE = False

    def get_model_matrix(self):
        self.scale = CorrelationEditor.WORLD_PIXEL_SIZE / self.pixel_size
        scale_mat = self.scale * np.identity(4)
        scale_mat[3, 3] = 1.0

        rotation_mat = np.identity(4)
        _cos = np.cos(self.rotation * 180.0 / np.pi)
        _sin = np.sin(self.rotation * 180.0 / np.pi)
        rotation_mat[0, 0] = _cos
        rotation_mat[1, 0] = _sin
        rotation_mat[0, 1] = -_sin
        rotation_mat[1, 1] = _cos

        translation_mat = np.identity(4)
        translation_mat[0, 3] = self.translation[0]
        translation_mat[1, 3] = self.translation[1]

        parity_mat = np.identity(4)
        parity_mat[0, 0] = -1.0 if self.flip else 1.0

        return translation_mat * (rotation_mat * (scale_mat * parity_mat))

    def update_lut(self):
        if self.lut > 0:
            lut_array = np.asarray(settings.luts[settings.lut_names[self.lut - 1]])
        else:
            lut_array = np.asarray(settings.luts[settings.lut_names[0]]) * np.asarray(self.colour[0:3])
        if lut_array.shape[1] == 3:
            lut_array = np.reshape(lut_array, (1, lut_array.shape[0], 3))
            self.lut_texture.update(lut_array)

    def compute_autocontrast(self):
        subsample = self.data[::settings.autocontrast_subsample, ::settings.autocontrast_subsample]
        n = subsample.shape[0] * subsample.shape[1]
        sorted_pixelvals = np.sort(subsample.flatten())
        self.contrast_lims[0] = sorted_pixelvals[int(settings.autocontrast_saturation / 100.0 * n)]
        self.contrast_lims[1] = sorted_pixelvals[int((1.0 - settings.autocontrast_saturation / 100.0) * n)]

    def compute_histogram(self):
        self.hist_vals, self.hist_bins = np.histogram(self.data, bins=CorrelationEditor.HISTOGRAM_BINS)
        self.hist_vals = self.hist_vals.astype('float32')
        self.hist_bins = self.hist_bins.astype('float32')
        self.hist_vals = np.delete(self.hist_vals, 0)
        self.hist_bins = np.delete(self.hist_bins, 0)

    def generate_va(self):
        # set up the quad vertex array
        w, h = self.width * 0.5, self.height * 0.5
        vertex_attributes = [-w, h, 1.0, 0.0, 1.0,
                             -w, -h, 1.0, 0.0, 0.0,
                             w, -h, 1.0, 1.0, 0.0,
                             w, h, 1.0, 1.0, 1.0]
        indices = [0, 1, 2, 2, 0, 3]
        self.quad_va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))

        # set up the border vertex array
        vertex_attributes = [-w, h,
                             w, h,
                             w, -h,
                             -w, -h]
        indices = [0, 1, 1, 2, 2, 3, 3, 0]
        self.border_va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))
