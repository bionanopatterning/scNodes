import imgui
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
from opengl_classes import *
import settings
import time
from itertools import count

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

        COLOUR_IMAGE_BORDER = (1.0, 0.8, 0.2, 1.0)


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

    frames = list()  # order of frames in this list determines the rendering order. Index 0 = front, index -1 = back.
    active_frame = None

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

        self.renderer = Renderer()
        self.camera = Camera()

        ## DEBUG
        CorrelationEditor.frames.append(CLEMFrame(np.random.randint(0, 255, (1024, 1024))))
        CorrelationEditor.active_frame = CorrelationEditor.frames[0]

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
        self.camera_control()
        self.camera.on_update()
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
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *CorrelationEditor.COLOUR_FRAME_DARK)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *CorrelationEditor.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB, *CorrelationEditor.COLOUR_FRAME_DARK)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB_HOVERED, *CorrelationEditor.COLOUR_FRAME_DARK)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB_ACTIVE, *CorrelationEditor.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_BACKGROUND, *CorrelationEditor.COLOUR_FRAME_EXTRA_DARK)
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *CorrelationEditor.COLOUR_TEXT)
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, *CorrelationEditor.COLOUR_TEXT)
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *CorrelationEditor.COLOUR_FRAME_EXTRA_DARK)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, CorrelationEditor.WINDOW_ROUNDING)

        for frame in CorrelationEditor.frames:
            self.renderer.render_frame(self.camera, frame)

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
        imgui.begin("Selected frame parameters", False, imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_COLLAPSE)
        #  Transform info
        expanded, _ = imgui.collapsing_header("Transform", None)
        if expanded:
            imgui.push_item_width(50)
            _, af.translation[0] = imgui.input_float("x", af.translation[0], CorrelationEditor.TRANSLATION_INCREMENT, CorrelationEditor.TRANSLATION_INCREMENT_LARGE, '%.1f nm', imgui.INPUT_TEXT_AUTO_SELECT_ALL)
            imgui.same_line(spacing=20)
            _, af.translation[1] = imgui.input_float("y", af.translation[0], CorrelationEditor.TRANSLATION_INCREMENT, CorrelationEditor.TRANSLATION_INCREMENT_LARGE, '%.1f nm', imgui.INPUT_TEXT_AUTO_SELECT_ALL)
            _, af.rotation = imgui.input_float("rotation", af.rotation, CorrelationEditor.ROTATION_INCREMENT, CorrelationEditor.ROTATION_INCREMENT_LARGE, '%.2f°', imgui.INPUT_TEXT_AUTO_SELECT_ALL)
            _, af.pixel_size = imgui.input_float("pixel size (nm)", af.pixel_size, CorrelationEditor.PIXEL_SIZE_INCREMENT, CorrelationEditor.PIXEL_SIZE_INCREMENT_LARGE, '%.1f', imgui.INPUT_TEXT_AUTO_SELECT_ALL)
            _, af.flip = imgui.checkbox("flip", af.flip)
            imgui.pop_item_width()
            imgui.separator()

        # Visuals info
        expanded, _ = imgui.collapsing_header("Visuals", None, imgui.TREE_NODE_DEFAULT_OPEN)
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
            imgui.same_line(spacing=-1 if (af.lut == 0) else 35)
            if imgui.button("A", 19, 19):
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

    def objects_info_window(self):
        pass # TODO

    def visuals_window(self):
        pass # TODO

    def camera_control(self):
        pass # TODO

    def end_frame(self):
        self.window.end_frame()

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


class Renderer:

    def __init__(self):
        self.quad_shader = Shader("shaders/ce_quad_shader.glsl")
        self.border_shader = Shader("shaders/ce_border_shader.glsl")

    def render_frame(self, camera, frame):
        pass
        # TODO: figure out how to implement z-ordering. 1) Give quads a z coordinate, or 2) disable depth testing and manually manage rendering order
        # TODO: figure out whether to render to screen directly, or to FBO which is rendered onto a quad on the screen.
        # FOR NOW: no FBO, and z ordering manually + update blend mode before each frame
        # set blend mode #TODO

        # render image
        self.quad_shader.bind()
        frame.quad_va.bind()
        frame.texture.bind(0)
        frame.lut_texture.bind(1)
        print("Rendering quad w image.")
        print(frame.contrast_lims)
        self.quad_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.quad_shader.uniform2f("contrastLimits", frame.contrast_lims)
        self.quad_shader.uniformmat4("modelMatrix", frame.get_model_matrix()) # TODO: get model matrix not just for frame itself, but rather product of all the frame's parent's transforms.
        glDrawElements(GL_TRIANGLES, frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.quad_shader.unbind()
        frame.quad_va.unbind()
        glActiveTexture(GL_TEXTURE0)

        # if frame is the active one, render border.
        if CorrelationEditor.active_frame == frame:
            self.border_shader.bind()
            frame.border_va.bind()
            self.border_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
            self.border_shader.uniformmat4("modelMatrix", frame.get_model_matrix())
            self.border_shader.uniform3f("lineColour", CorrelationEditor.COLOUR_IMAGE_BORDER)
            glDrawElements(GL_LINES, frame.border_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            self.border_shader.unbind()
            frame.border_va.unbind()


class Camera:
    def __init__(self):
        self.view_matrix = np.identity(4)
        self.projection_matrix = np.identity(4)
        self.view_projection_matrix = np.identity(4)
        self.position = np.zeros(3)
        self.zoom = 1.0
        self.set_projection_matrix(settings.ne_window_width, settings.ne_window_height)

    def set_projection_matrix(self, window_width, window_height):
        self.projection_matrix = np.matrix([
            [2 / window_width, 0, 0, 0],
            [0, 2 / window_height, 0, 0],
            [0, 0, -2 / 100, 0],
            [0, 0, 0, 1],
        ])

    def on_update(self):
        self.view_matrix = np.matrix([
            [self.zoom, 0.0, 0.0, self.position[0]],
            [0.0, self.zoom, 0.0, self.position[1]],
            [0.0, 0.0, self.zoom, self.position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.view_projection_matrix = np.matmul(self.projection_matrix, self.view_matrix)


class CLEMFrame:
    idgen = count(0)

    def __init__(self, img_array):
        """Grayscale images only - img_array must be a 2D np.ndarray"""

        # data
        self.uid = next(CLEMFrame.idgen)
        self.data = img_array
        self.width, self.height = self.data.shape
        self.children = list()
        self.parent = None

        # transform parameters
        self.pixel_size = 100.0  # pixel size in nm
        self.pivot_point = np.zeros(2)  # pivot point for rotation and scaling of this particular image. can be moved by the user. In _local coordinates_, i.e. relative to where the frame itself is positioned.
        self.translation = np.zeros(2)
        self.rotation = 0.0
        self.scale = 1.0  # automatically updated in get_model_matrix. Internal var to keep track of relative scale of quad.
        self.flip = False  # 1.0 or -1.0, in case the image is mirrored.

        # visuals
        self.blend_mode = CorrelationEditor.BLEND_MODES[0]
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
        self.quad_va = VertexArray()
        self.border_va = VertexArray(attribute_format="xy")
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
        print(rotation_mat)
        # TODO fix matrices here, something's going wrong in rendering

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

    def move_to_front(self):
        CorrelationEditor.frames.insert(0, CorrelationEditor.frames.pop(CorrelationEditor.frames.index(self)))

    def move_to_back(self):
        CorrelationEditor.frames.append(CorrelationEditor.frames.pop(CorrelationEditor.frames.index(self)))

    def move_backwards(self):
        idx = CorrelationEditor.frames.index(self)
        if idx < (len(CorrelationEditor.frames) - 1):
            CorrelationEditor.frames[idx], CorrelationEditor.frames[idx + 1] = CorrelationEditor.frames[idx + 1], CorrelationEditor.frames[idx]

    def move_forwards(self):
        idx = CorrelationEditor.frames.index(self)
        if idx > 0:
            CorrelationEditor.frames[idx], CorrelationEditor.frames[idx - 1] = CorrelationEditor.frames[idx - 1], CorrelationEditor.frames[idx]

    def __eq__(self, other):
        if isinstance(other, CLEMFrame):
            return self.uid == other.uid
        return False