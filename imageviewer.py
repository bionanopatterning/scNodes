import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from opengl_classes import *
import settings
import config as cfg
from dataset import *
from roi import *


class ImageViewer:
    NO_IMAGE_NOTIFICATION_SIZE = (300, 20)
    MARKER_SIZE = 4.5
    MARKER_COLOUR = (230 / 255, 174 / 255, 13 / 255, 1.0)

    COLOUR_FRAME_BACKGROUND = (0.24, 0.24, 0.24)
    COLOUR_MAIN = (0.3 / 1.3, 0.3 / 1.3, 0.3 / 1.3)
    COLOUR_THEME_ACTIVE = (0.59, 1.0, 0.86)
    COLOUR_THEME = (0.729, 1.0, 0.95)
    COLOUR_MAIN_BRIGHT = (0.3 / 1, 0.3 / 1, 0.3 / 1)
    COLOUR_MAIN_DARK = (0.3 / 1.5, 0.3 / 1.5, 0.3 / 1.5)
    COLOUR_WINDOW_BACKGROUND = (0.14, 0.14, 0.14, 0.8)
    COLOUR_CLEAR = (0.94, 0.94, 0.94, 1.0)

    CONTRAST_WINDOW_SIZE = [200, 240]
    CONTEXT_MENU_SIZE = [200, 100]
    INFO_BAR_HEIGHT = 100

    CAMERA_PAN_SPEED = 1.0
    CAMERA_ZOOM_MAX = 50
    CAMERA_ZOOM_STEP = 0.1

    HISTOGRAM_BINS = 50
    AUTOCONTRAST_SUBSAMPLE = 2
    AUTOCONTRAST_SATURATE = 0.3

    FRAME_SELECT_BUTTON_SIZE = [20, 19]
    FRAME_SELECT_BUTTON_SPACING = 4

    def __init__(self, window, shared_font_atlas=None):
        # glfw and imgui
        self.window = window
        self.window.make_current()

        if shared_font_atlas is not None:
            self.imgui_context = imgui.create_context(shared_font_atlas)
        else:
            imgui.get_current_context()
            self.imgui_context = imgui.create_context()
        self.imgui_implementation = GlfwRenderer(self.window.glfw_window)
        self.window.set_callbacks()
        self.window.set_window_callbacks()
        self.window.clear_color = ImageViewer.COLOUR_CLEAR
        # Rendering related objects and vars
        self.shader = Shader("shaders/textured_shader.glsl")
        self.roi_shader = Shader("shaders/line_shader.glsl")
        self.texture = Texture(format="rgb32f")
        self.fbo = FrameBuffer(*settings.def_img_size)
        self.va = VertexArray()
        self.lut_texture = Texture(format="rgb32f")
        self.current_lut = 0
        self.lut_array = None
        self.set_lut(self.current_lut)
        self.camera = Camera()

        # GUI vars
        self.autocontrast = [True, True, True]
        self.contrast_min = [0, 0, 0]
        self.contrast_max = [65535, 65535, 65535]

        self.contrast_window_open = False
        self.contrast_window_channel = 0
        self.contrast_window_position = [0, 0]
        self.context_menu_open = False
        self.context_menu_can_close = False
        self.context_menu_position = [0, 0]

        # Change flags
        self.image_size_changed = False
        self.image_requires_update = False
        self.previous_active_node = None
        self.new_image_requested = False
        # Image data
        self.original_image = None
        self.image = None
        self.image_pxd = None
        self.image_width = None
        self.image_height = None
        self.image_amax = [0, 0, 0]
        self.hist_counts = [list(), list(), list()]
        self.hist_bins = [list(), list(), list()]
        self.mode = "R"
        self.set_image(np.zeros((16, 16)))
        self.show_image = False

        # ROI data - to be made compatible with NodeEditor later
        self.drawing_roi = False
        self.moving_roi = False
        self.roi = ROI([0, 0, 0, 0], (1.0, 0.0, 1.0, 1.0))
        self.marker = Marker([-ImageViewer.MARKER_SIZE, 0.0, ImageViewer.MARKER_SIZE, 0.0, 0.0,-ImageViewer.MARKER_SIZE, 0.0, ImageViewer.MARKER_SIZE], [0, 1, 2, 3], ImageViewer.MARKER_COLOUR)

        # GUI behaviour
        self.show_frame_select_window = True
        #
        self.current_dataset = Dataset()
        self.frame_info = ""

    def set_mode(self, mode):
        if mode in ["R", "RGB"]:
            self.mode = mode
            self.show_frame_select_window = self.mode == "R"
        else:
            print(f"ImageViewer.set_mode with mode = {mode} is not a valid mode!")

    def on_update(self):
        imgui.set_current_context(self.imgui_context)
        self.window.make_current()
        self.window.set_full_viewport()
        if self.window.focused:
            self.imgui_implementation.process_inputs()
        if self.window.window_gained_focus:
            self.window.pop_any_mouse_event()

        self.window.on_update()
        self.imgui_implementation.refresh_font_texture()
        if self.window.window_size_changed:
            settings.iv_window_height = self.window.height
            settings.iv_window_width = self.window.width
            self.camera.set_projection_matrix(self.window.width, self.window.height)
            self.window.window_size_changed = False

        if self.image_requires_update:
            self.update_image()
            self.image_requires_update = False

        if self.image_size_changed:
            self.image_size_changed = False
            self.center_camera()

        self.camera.on_update()
        self._camera_control()
        self._render()
        self._edit_and_render_roi()

        # imgui
        imgui.new_frame()
        # Push overall imgui style vars
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *ImageViewer.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *ImageViewer.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *ImageViewer.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *ImageViewer.COLOUR_MAIN_DARK)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *ImageViewer.COLOUR_MAIN_BRIGHT)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *ImageViewer.COLOUR_THEME)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *ImageViewer.COLOUR_THEME_ACTIVE)
        imgui.push_style_color(imgui.COLOR_BUTTON, *ImageViewer.COLOUR_MAIN)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *ImageViewer.COLOUR_THEME_ACTIVE)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *ImageViewer.COLOUR_MAIN)
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *ImageViewer.COLOUR_THEME)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *ImageViewer.COLOUR_WINDOW_BACKGROUND)

        self._shortcuts()
        self._gui_main()

        imgui.pop_style_color(12)
        imgui.render()
        self.imgui_implementation.render(imgui.get_draw_data())

        if cfg.active_node is not None:
            ImageViewer.MARKER_COLOUR = cfg.active_node.colour
            if self.previous_active_node is not cfg.active_node or cfg.active_node.any_change or self.new_image_requested:
                if cfg.active_node.returns_image:
                    an = cfg.active_node
                    self.current_dataset = an.get_source_load_data_node(an).dataset
                    if self.current_dataset.initialized:
                        self.current_dataset.current_frame = np.clip(self.current_dataset.current_frame, 0, self.current_dataset.n_frames - 1)
                        cfg.active_node.frame_requested_by_image_viewer = True
                        new_image = cfg.active_node.get_image(self.current_dataset.current_frame)
                        cfg.active_node.frame_requested_by_image_viewer = False
                        if new_image is not None:
                            self.show_image = True
                            self.set_image(new_image)
                        else:
                            self.show_image = False
                    else:
                        self.show_image = False
            self.previous_active_node = cfg.active_node

    def end_frame(self):
        self.window.end_frame()

    def close(self):
        self.imgui_implementation.shutdown()

    def _gui_main(self):
        if self.show_frame_select_window and self.current_dataset.initialized:
            self._frame_info_window()

        # Context menu
        if not self.context_menu_open:
            if self.window.get_mouse_event(glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0):
                self.context_menu_position = self.window.cursor_pos
                self.context_menu_open = True
                self.context_menu_can_close = False
        else:
            self._context_menu()

        # Brightness/contrast window
        if self.contrast_window_open:
            imgui.set_next_window_size(ImageViewer.CONTRAST_WINDOW_SIZE[0], ImageViewer.CONTRAST_WINDOW_SIZE[1])
            imgui.set_next_window_position(self.contrast_window_position[0], self.contrast_window_position[1], condition = imgui.APPEARING)

            _, self.contrast_window_open = imgui.begin("Adjust contrast", closable = True, flags = imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE)
            self.contrast_window_position = imgui.get_window_position()
            available_width = imgui.get_content_region_available_width()
            _rgb_button_spacing = 4

            if self.mode == "RGB":
                if imgui.button("R", width=(available_width - 2 * _rgb_button_spacing) / 3, height=30):
                    self.contrast_window_channel = 0
                imgui.same_line(spacing = _rgb_button_spacing)
                if imgui.button("G", width=(available_width - 2 * _rgb_button_spacing) / 3, height=30):
                    self.contrast_window_channel = 1
                imgui.same_line(spacing=_rgb_button_spacing)
                if imgui.button("B", width=(available_width - 2 * _rgb_button_spacing) / 3, height=30):
                    self.contrast_window_channel = 2

            if self.mode == "R":
                self.contrast_window_channel = 0
                _idx = int(self.lut_array.shape[0] / 2)
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *self.lut_array[-_idx, :])
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, *self.lut_array[-_idx, :])
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *self.lut_array[-_idx, :])
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *self.lut_array[-_idx, :])
            else:
                _clr = (float(self.contrast_window_channel == 0) + 0.2 * 0.8, float(self.contrast_window_channel == 1) + 0.2 * 0.8, float(self.contrast_window_channel == 2) + 0.2 * 0.8)
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *_clr)
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, *_clr)
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *_clr)
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *_clr)


            imgui.plot_histogram("##contrast_hist", self.hist_counts[self.contrast_window_channel], graph_size = (available_width, 100))
            imgui.push_item_width(available_width)
            _min_changed, self.contrast_min[self.contrast_window_channel] = imgui.slider_float("##min_contrast", self.contrast_min[self.contrast_window_channel], 0, self.image_amax[self.contrast_window_channel] + 1, format="min: %1.0f")
            _max_changed, self.contrast_max[self.contrast_window_channel] = imgui.slider_float("##max_contrast", self.contrast_max[self.contrast_window_channel], 0, self.image_amax[self.contrast_window_channel] + 1, format="max: %1.0f")
            if _min_changed or _max_changed:
                self.autocontrast[self.contrast_window_channel] = False
            imgui.pop_item_width()
            if imgui.button("Auto once", width = 80, height = ImageViewer.FRAME_SELECT_BUTTON_SIZE[1]):
                self._compute_auto_contrast(channel = self.contrast_window_channel)
            imgui.same_line(spacing = ImageViewer.FRAME_SELECT_BUTTON_SPACING)
            _always_auto_changed, self.autocontrast[self.contrast_window_channel] = imgui.checkbox("always auto", self.autocontrast[self.contrast_window_channel])
            if _always_auto_changed and self.autocontrast[self.contrast_window_channel]:
                self._compute_auto_contrast()
            if imgui.is_window_focused():
                if self.window.get_key_event(glfw.KEY_SPACE, glfw.PRESS):
                    self._compute_auto_contrast()
            imgui.pop_style_color(4)
            imgui.end()

        # No image provided by active node
        if self.show_image is False:
            imgui.set_next_window_position(self.window.width // 2 - ImageViewer.NO_IMAGE_NOTIFICATION_SIZE[0] // 2, self.window.height // 2 - ImageViewer.NO_IMAGE_NOTIFICATION_SIZE[1] // 2)
            imgui.set_next_window_size(*ImageViewer.NO_IMAGE_NOTIFICATION_SIZE)
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *ImageViewer.COLOUR_CLEAR)
            imgui.push_style_color(imgui.COLOR_BORDER, *ImageViewer.COLOUR_CLEAR)
            imgui.push_style_color(imgui.COLOR_TEXT, *(0.0, 0.0, 0.0, 1.0))
            imgui.begin("##", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_COLLAPSE)
            imgui.text("   Active node does not output an image.")
            imgui.end()
            imgui.pop_style_color(3)

    def _render(self):
        if self.show_image:
            self.va.bind()
            self.shader.bind()
            self.texture.bind(0)
            self.lut_texture.bind(1)
            self.shader.uniformmat4("cameraMatrix", self.camera.view_projection_matrix)
            self.shader.uniform3f("contrast_min", self.contrast_min)
            self.shader.uniform3f("contrast_max", self.contrast_max)
            self.shader.uniform3f("translation", [0.0, 0.0, 0.0])
            self.shader.uniform1i("use_lut", 1 if self.mode == "R" else 0)
            glDrawElements(GL_TRIANGLES, self.va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            self.shader.unbind()
            self.va.unbind()
            glActiveTexture(GL_TEXTURE0)
            self.marker.render_start(self.roi_shader, self.camera, ImageViewer.MARKER_COLOUR)
            for coordinate in self.image.maxima:
                translation = [coordinate[1], coordinate[0]]
                self.marker.render(self.roi_shader, translation)
            self.marker.render_end(self.roi_shader)

    def _frame_info_window(self):
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)
        imgui.set_next_window_position(0, self.window.height - ImageViewer.INFO_BAR_HEIGHT + (cfg.window_height - self.window.height), imgui.ALWAYS)
        imgui.set_next_window_size(self.window.width, ImageViewer.INFO_BAR_HEIGHT)
        ## Info & control panel
        imgui.begin("##frameselectwindow", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)

        # Frame slider and buttons
        available_width = imgui.get_content_region_available_width()
        _frame_idx_changed = False
        if imgui.button("<", width=ImageViewer.FRAME_SELECT_BUTTON_SIZE[0], height=ImageViewer.FRAME_SELECT_BUTTON_SIZE[1]):
            self.current_dataset.current_frame -= 1
            _frame_idx_changed = True
        imgui.same_line(spacing=ImageViewer.FRAME_SELECT_BUTTON_SPACING)
        imgui.push_item_width(
            available_width - 2 * ImageViewer.FRAME_SELECT_BUTTON_SIZE[0] - 2 * ImageViewer.FRAME_SELECT_BUTTON_SPACING)

        _slider_changed, self.current_dataset.current_frame = imgui.slider_int("##frame_select",
                                                                               self.current_dataset.current_frame, 0,
                                                                               self.current_dataset.n_frames - 1,
                                                                               f"Frame {self.current_dataset.current_frame}/{self.current_dataset.n_frames}")

        _frame_idx_changed = _slider_changed or _frame_idx_changed
        imgui.same_line(spacing=ImageViewer.FRAME_SELECT_BUTTON_SPACING)
        if imgui.button(">", width=ImageViewer.FRAME_SELECT_BUTTON_SIZE[0], height=ImageViewer.FRAME_SELECT_BUTTON_SIZE[1]):
            self.current_dataset.current_frame += 1
            _frame_idx_changed = True
        if not self.window.get_key(glfw.KEY_LEFT_SHIFT):
            if self.window.scroll_delta[1] != 0:
                _frame_idx_changed = True
                self.current_dataset.current_frame -= int(self.window.scroll_delta[1])
        if self.window.get_key_event(glfw.KEY_LEFT, glfw.PRESS, mods=0) or self.window.get_key_event(glfw.KEY_LEFT,
                                                                                                     glfw.REPEAT,
                                                                                                     mods=0):
            _frame_idx_changed = True
            self.current_dataset.current_frame -= 1
        if self.window.get_key_event(glfw.KEY_RIGHT, glfw.PRESS, mods=0) or self.window.get_key_event(glfw.KEY_RIGHT,
                                                                                                      glfw.REPEAT,
                                                                                                      mods=0):
            _frame_idx_changed = True
            self.current_dataset.current_frame += 1
        self.new_image_requested = False
        if _frame_idx_changed:
            self.current_dataset.current_frame = np.clip(self.current_dataset.current_frame, 0,
                                                         self.current_dataset.n_frames - 1)
            self.new_image_requested = True

        imgui.pop_item_width()

        # Image info
        imgui.separator()
        imgui.text(self.frame_info)
        imgui.pop_style_var(1)
        imgui.end()

    def _edit_and_render_roi(self):
        if not self.show_image:
            return None
        if cfg.active_node is None:
            return None
        if not cfg.active_node.use_roi:
            return None
        # if past the above, active node has and used a roi.
        if cfg.active_node.roi == [0, 0, 0, 0]:
            cfg.active_node.roi = [self.image_width // 4, self.image_height // 4, self.image_width * 3 // 4, self.image_height * 3 // 4]
        else:
            self.roi.set_box(cfg.active_node.roi)
            self.roi.colour = cfg.active_node.colour
            self.roi.render(self.roi_shader, self.camera)
            # edit roi
            if not imgui.get_io().want_capture_mouse and not self.window.window_gained_focus:
                if self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.PRESS):
                    # get world position of mouse click.
                    px_coords = self.get_cursor_image_coordinates()
                    if self.roi.is_in_roi(px_coords):
                        self.moving_roi = True
                    else:
                        self.drawing_roi = True
                        self.roi.set_box([*px_coords, *px_coords])

                if self.moving_roi:
                    if self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE):
                        self.moving_roi = False
                        self.roi.limit(self.image_width, self.image_height)
                        cfg.active_node.any_change = True
                    else:
                        self.roi.translate(self.cursor_delta_as_world_delta())

                elif self.drawing_roi:
                    px_coords = self.get_cursor_image_coordinates()
                    if self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE):
                        self.drawing_roi = False
                        self.roi.correct_order()
                        self.roi.limit(self.image_width, self.image_height)
                        cfg.active_node.any_change = True
                    else:
                        new_box = [self.roi.box[0], self.roi.box[1], px_coords[0], px_coords[1]]
                        self.roi.set_box(new_box)

            cfg.active_node.roi = self.roi.box

    def _camera_control(self):
        if not imgui.get_io().want_capture_mouse:
            if self.window.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE):
                self.camera.position[0] += self.window.cursor_delta[0] * ImageViewer.CAMERA_PAN_SPEED
                self.camera.position[1] -= self.window.cursor_delta[1] * ImageViewer.CAMERA_PAN_SPEED
            if self.window.get_key(glfw.KEY_LEFT_SHIFT):
                if self.window.scroll_delta[1] != 0:
                    camera_updated_zoom = self.camera.zoom * (1.0 + self.window.scroll_delta[1] * ImageViewer.CAMERA_ZOOM_STEP)
                    if camera_updated_zoom < ImageViewer.CAMERA_ZOOM_MAX:
                        self.camera.zoom *= (1.0 + self.window.scroll_delta[1] * ImageViewer.CAMERA_ZOOM_STEP)
                        self.camera.position[0] *= (1.0 + self.window.scroll_delta[1] * ImageViewer.CAMERA_ZOOM_STEP)
                        self.camera.position[1] *= (1.0 + self.window.scroll_delta[1] * ImageViewer.CAMERA_ZOOM_STEP)

    def _shortcuts(self):
        if self.window.get_key_event(glfw.KEY_C, glfw.PRESS, glfw.MOD_SHIFT | glfw.MOD_CONTROL):
            self.contrast_window_open = True
        if self.window.get_key_event(glfw.KEY_W, glfw.PRESS, glfw.MOD_CONTROL):
            self.contrast_window_open = False
        if self.contrast_window_open:
            if self.window.get_key_event(glfw.KEY_SPACE, glfw.PRESS):
                self._compute_auto_contrast(self.contrast_window_channel)

    def _context_menu(self):
        imgui.set_next_window_position(self.context_menu_position[0] - 3, self.context_menu_position[1] - 3)
        imgui.set_next_window_size(ImageViewer.CONTEXT_MENU_SIZE[0], ImageViewer.CONTEXT_MENU_SIZE[1])
        imgui.begin("##ivcontextmenu", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE)
        # Close context menu when it is not hovered.
        context_menu_hovered = imgui.is_window_hovered(flags=imgui.HOVERED_ALLOW_WHEN_BLOCKED_BY_POPUP | imgui.HOVERED_CHILD_WINDOWS)
        if context_menu_hovered:
            self.context_menu_can_close = True
        if not context_menu_hovered and self.context_menu_can_close:
            self.context_menu_can_close = False
            self.context_menu_open = False

        _open, _ = imgui.menu_item("Brightness / contrast")
        if _open:
            self.contrast_window_open = True
            self.contrast_window_position = self.context_menu_position
        _reset_view, _ = imgui.menu_item("Reset view")
        if _reset_view:
            self.center_camera()
            self.context_menu_open = False
        if imgui.begin_menu("Change LUT"):
            for key in settings.lut_names:
                key_clicked, _ = imgui.menu_item(key)
                if key_clicked:
                    self.set_lut(settings.lut_names.index(key))
            imgui.end_menu()
        imgui.end()

    def set_image( self, image):
        self.image_requires_update = True
        if type(image) == np.ndarray:
            self.image = Frame("virtual_path")
            self.image.data = image
        else:
            self.image = image
        self.frame_info = str(self.image)

    def update_image(self):
        self.image_pxd = self.image.load()
        width = np.shape(self.image_pxd)[1]
        height = np.shape(self.image_pxd)[0]
        if len(self.image_pxd.shape) == 2:
            self.image_pxd = np.repeat(self.image_pxd[:, :, np.newaxis], 3, axis=2)
            self.set_mode("R")
        else:
            self.set_mode("RGB")
        self.image_amax = np.amax(self.image_pxd, axis = (0, 1))
        if self.autocontrast[0]:
            self._compute_auto_contrast(0)
        if self.autocontrast[1]:
            self._compute_auto_contrast(1)
        if self.autocontrast[1]:
            self._compute_auto_contrast(2)
        if self.image_width != width or self.image_height != height:
            self.image_size_changed = True
            self.image_width = width
            self.image_height = height

            vertex_attributes = [0.0, height, 1.0, 0.0, 1.0,
                                 0.0, 0.0, 1.0, 0.0, 0.0,
                                 width, 0.0, 1.0, 1.0, 0.0,
                                 width, height, 1.0, 1.0, 1.0]
            indices = [0, 1, 2, 2, 0, 3]
            self.va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))
        self.texture.update(self.image_pxd.astype(np.float))

        self.hist_counts[0], self.hist_bins[0] = np.histogram(self.image_pxd[:, :, 0], bins=ImageViewer.HISTOGRAM_BINS)
        self.hist_counts[0] = self.hist_counts[0].astype('float32')
        self.hist_counts[0] = np.delete(self.hist_counts[0], 0)
        self.hist_bins[0] = np.delete(self.hist_bins[0], 0)
        if self.mode == "RGB":
            self.hist_counts[1], self.hist_bins[1] = np.histogram(self.image_pxd[:, :, 1], bins=ImageViewer.HISTOGRAM_BINS)
            self.hist_counts[1] = self.hist_counts[1].astype('float32')
            self.hist_counts[1] = np.delete(self.hist_counts[1], 0)
            self.hist_bins[1] = np.delete(self.hist_bins[1], 0)

            self.hist_counts[2], self.hist_bins[2] = np.histogram(self.image_pxd[:, :, 2], bins=ImageViewer.HISTOGRAM_BINS)
            self.hist_counts[2] = self.hist_counts[2].astype('float32')
            self.hist_counts[2] = np.delete(self.hist_counts[2], 0)
            self.hist_bins[2] = np.delete(self.hist_bins[2], 0)

    def _compute_auto_contrast(self, channel = None):
        img_subsample = self.image_pxd[::ImageViewer.AUTOCONTRAST_SUBSAMPLE, ::ImageViewer.AUTOCONTRAST_SUBSAMPLE, :]
        N = img_subsample.shape[0] * img_subsample.shape[1]
        for i in range(3):
            if self.autocontrast[i] or channel == i:
                img_sorted = np.sort(img_subsample[:, :, i].flatten())
                self.contrast_min[i] = img_sorted[int(ImageViewer.AUTOCONTRAST_SATURATE / 100.0 * N)]
                self.contrast_max[i] = img_sorted[int((1.0 - ImageViewer.AUTOCONTRAST_SATURATE / 100.0) * N)]

    def center_camera(self):
        self.camera.zoom = 1
        self.camera.position = [-self.image_width / 2, -self.image_height / 2, 0.0]

    def set_lut(self, lut_index):
        self.current_lut = lut_index
        self.lut_array = np.asarray(settings.luts[settings.lut_names[self.current_lut]])
        if self.lut_array.shape[1] == 3:
            lut_array = np.reshape(self.lut_array, (self.lut_array.shape[0], 1, self.lut_array.shape[1]))
        self.lut_texture.update(lut_array)

    def get_cursor_image_coordinates(self):
        c_pos = self.window.cursor_pos  # cursor position
        c_pos_ndc = [(c_pos[0] / self.window.width) * 2.0 - 1.0, -((c_pos[1] / self.window.height) * 2.0 - 1.0)]
        world_pos = np.matmul(self.camera.view_matrix.I, self.camera.projection_matrix.I) \
                    * np.matrix([[c_pos_ndc[0]], [c_pos_ndc[1]], [1], [1]])
        image_coordinates = [int(world_pos[0]), int(world_pos[1])]
        return image_coordinates

    def cursor_delta_as_world_delta(self):
        return [self.window.cursor_delta[0] / self.camera.zoom, -self.window.cursor_delta[1] / self.camera.zoom]


class Camera:
    def __init__(self):
        self.position = np.asarray([0.0, 0.0, 0.0])
        self.zoom = 1.0
        self.view_matrix = np.identity(4)
        self.projection_matrix = np.identity(4)
        self.view_projection_matrix = np.identity(4)
        self.set_projection_matrix(settings.iv_window_width, settings.iv_window_height)

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



