import glfw
import imgui
import scNodes.core.config as cfg
import numpy as np
from itertools import count
import mrcfile
from scNodes.core.opengl_classes import *
import datetime


class SegmentationEditor:
    if True:
        CAMERA_ZOOM_STEP = 0.1
        CAMERA_MAX_ZOOM = 100.0
        DEFAULT_HORIZONTAL_FOV_WIDTH = 50000  # upon init, camera zoom is such that from left to right of window = 50 micron.
        DEFAULT_ZOOM = 1.0  # adjusted in init
        DEFAULT_WORLD_PIXEL_SIZE = 1.0  # adjusted on init

        # GUI params
        MAIN_WINDOW_WIDTH = 270
        FEATURE_PANEL_HEIGHT = 80

    def __init__(self, window, imgui_context, imgui_impl):
        self.window = window
        self.window.clear_color = cfg.COLOUR_WINDOW_BACKGROUND
        self.window.make_current()
        self.imgui_context = imgui_context
        self.imgui_implementation = imgui_impl

        self.camera = Camera()
        SegmentationEditor.DEFAULT_ZOOM = cfg.window_height / SegmentationEditor.DEFAULT_HORIZONTAL_FOV_WIDTH  # now it is DEFAULT_HORIZONTAL_FOV_WIDTH
        SegmentationEditor.DEFAULT_WORLD_PIXEL_SIZE = 1.0 / SegmentationEditor.DEFAULT_ZOOM
        self.camera.zoom = SegmentationEditor.DEFAULT_ZOOM
        self.renderer = Renderer()


        sef = SEFrame("C:/Users/mart_/Desktop/tomo.mrc")
        cfg.se_frames.append(sef)
        cfg.se_active_frame = sef

        sef.features.append(Segmentation(sef, "def"))

    def on_update(self):
        imgui.set_current_context(self.imgui_context)
        self.window.make_current()
        self.window.set_full_viewport()
        if self.window.focused:
            self.imgui_implementation.process_inputs()

        if not imgui.get_io().want_capture_keyboard and imgui.is_key_pressed(glfw.KEY_TAB):
            if imgui.is_key_down(glfw.KEY_LEFT_SHIFT):
                cfg.active_editor = (cfg.active_editor - 1) % len(cfg.editors)
            else:
                cfg.active_editor = (cfg.active_editor + 1) % len(cfg.editors)

        self.window.on_update()
        if self.window.window_size_changed:
            cfg.window_width = self.window.width
            cfg.window_height = self.window.height
            self.camera.set_projection_matrix(cfg.window_width, cfg.window_height)

        imgui.get_io().display_size = self.window.width, self.window.height
        imgui.new_frame()

        # GUI stuff
        self.camera_control()
        self.camera.on_update()
        self.gui_main()
        self.input()

        imgui.render()
        self.imgui_implementation.render(imgui.get_draw_data())

    def input(self):
        if imgui.get_io().want_capture_mouse or imgui.get_io().want_capture_keyboard:
            return
        # drawing
        f = cfg.se_active_frame.active_feature
        if f is not None:
            cursor_world_position = self.camera.cursor_to_world_position(self.window.cursor_pos)
            pixel_coordinate = f.parent.world_to_pixel_coordinate(cursor_world_position)

            if imgui.is_mouse_down(0):
                Brush.apply_circular(f, pixel_coordinate, value=1)
            elif imgui.is_mouse_down(1):
                Brush.apply_circular(f, pixel_coordinate, value=255)


    def gui_main(self):
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *cfg.COLOUR_PANEL_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *cfg.COLOUR_TITLE_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *cfg.COLOUR_TITLE_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *cfg.COLOUR_TITLE_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *cfg.COLOUR_TITLE_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *cfg.COLOUR_TITLE_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *cfg.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_FRAME_ACTIVE)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *cfg.COLOUR_FRAME_ACTIVE)
        imgui.push_style_color(imgui.COLOR_BUTTON, *cfg.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *cfg.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *cfg.COLOUR_FRAME_EXTRA_DARK)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *cfg.COLOUR_FRAME_DARK)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *cfg.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB, *cfg.COLOUR_FRAME_DARK)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB_HOVERED, *cfg.COLOUR_FRAME_DARK)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB_ACTIVE, *cfg.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_BACKGROUND, *cfg.COLOUR_FRAME_EXTRA_DARK)
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *cfg.COLOUR_TEXT)
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, *cfg.COLOUR_TEXT)
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_FRAME_EXTRA_DARK)
        imgui.push_style_color(imgui.COLOR_MENUBAR_BACKGROUND, *cfg.COLOUR_MAIN_MENU_BAR)
        imgui.push_style_color(imgui.COLOR_HEADER, *cfg.COLOUR_HEADER)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *cfg.COLOUR_HEADER_HOVERED)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *cfg.COLOUR_HEADER_ACTIVE)
        imgui.push_style_color(imgui.COLOR_DRAG_DROP_TARGET, *cfg.COLOUR_DROP_TARGET)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, cfg.WINDOW_ROUNDING)

        def datasets_panel():
            if imgui.begin_child("dataset", 0.0, 80, True, imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR):
                for s in cfg.se_frames:
                    imgui.push_id(f"se{s.uid}")
                    if imgui.selectable(s.title, cfg.se_active_frame == s):
                        cfg.se_active_frame = s
                        imgui.pop_id()
                imgui.end_child()

        def features_panel():

            features = cfg.se_active_frame.features
            for f in features:
                pop_active_colour = False
                if cfg.se_active_frame.active_feature == f:
                    imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *cfg.COLOUR_FRAME_ACTIVE)
                    pop_active_colour = True
                if imgui.begin_child(f"##{f.uid}", 0.0, SegmentationEditor.FEATURE_PANEL_HEIGHT, True):
                    cw = imgui.get_content_region_available_width()

                    # Colour picker
                    _, f.colour = imgui.color_edit3(f.title, *f.colour[:3], imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)

                    # Title
                    imgui.same_line()
                    imgui.set_next_item_width(cw - 25)
                    _, f.title = imgui.input_text("##title", f.title, 256, imgui.INPUT_TEXT_NO_HORIZONTAL_SCROLL)

                    # Alpha slider and brush size
                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                    imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                    imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                    imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                    imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                    imgui.set_next_item_width(cw - 40)
                    pxs = cfg.se_active_frame.pixel_size
                    _, f.brushsize = imgui.slider_float("brush", f.brushsize, 0.1, 25.0, format="%.1f nm")
                    imgui.set_next_item_width(cw - 40)
                    _, f.alpha = imgui.slider_float("alpha", f.alpha, 0.0, 1.0, format="%.2f")
                    imgui.pop_style_var(5)

                    if imgui.is_window_hovered() and imgui.is_mouse_clicked(0):
                        cfg.se_active_frame.active_feature = f

                    if pop_active_colour:
                        imgui.pop_style_color(1)
                    imgui.end_child()

            # 'Add feature' button
            cw = imgui.get_content_region_available_width()
            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)

            imgui.new_line()
            imgui.same_line(spacing = (cw - 120) / 2)
            if imgui.button("Add feature", 120, 23):
                cfg.se_active_frame.features.append(Segmentation(cfg.se_active_frame, "Unnamed feature"))
            imgui.pop_style_var(1)

        def filters_panel():
            cw = imgui.get_content_region_available_width()
            imgui.push_item_width(cw)
            _, cfg.se_active_frame.contrast_lims[0] = imgui.drag_float("##min", cfg.se_active_frame.contrast_lims[0],
                                                                       1.0, 0.0, 0.0, format="min: %.0f")
            _, cfg.se_active_frame.contrast_lims[1] = imgui.drag_float("##max", cfg.se_active_frame.contrast_lims[1],
                                                                       1.0, 0.0, 0.0, format="max: %.0f")
            imgui.pop_item_width()

        # render the active frame
        if cfg.se_active_frame is not None:
            self.renderer.render_frame(cfg.se_active_frame, self.camera)

        imgui.set_next_window_position(0, 18, imgui.ONCE)
        imgui.set_next_window_size(SegmentationEditor.MAIN_WINDOW_WIDTH, self.window.height - 18)
        imgui.begin("##se_main", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        if imgui.collapsing_header("Available datasets", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            datasets_panel()
        if imgui.collapsing_header("Filters", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            filters_panel()
        if imgui.collapsing_header("Features", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            features_panel()

        imgui.end()

        imgui.pop_style_color(28)
        imgui.pop_style_var(1)

    def camera_control(self):
        if imgui.get_io().want_capture_mouse or imgui.get_io().want_capture_keyboard:
            return None
        if self.window.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE):
            delta_cursor = self.window.cursor_delta
            self.camera.position[0] += delta_cursor[0] / self.camera.zoom
            self.camera.position[1] -= delta_cursor[1] / self.camera.zoom
        if self.window.get_key(glfw.KEY_LEFT_SHIFT):
            self.camera.zoom *= (1.0 + self.window.scroll_delta[1] * SegmentationEditor.CAMERA_ZOOM_STEP)
            self.camera.zoom = min([self.camera.zoom, SegmentationEditor.CAMERA_MAX_ZOOM])

    def end_frame(self):
        self.window.end_frame()



class SEFrame:
    idgen = count(0)

    def __init__(self, path):
        uid_counter = next(SEFrame.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+"000") + uid_counter
        self.path = path
        self.title = os.path.splitext(os.path.basename(self.path))[0]
        self.n_slices = 0
        self.current_slice = -1
        self.data = None
        self.features = list()
        self.active_feature = None
        self.height, self.width = mrcfile.mmap(self.path, mode="r").data.shape[1:3]
        self.pixel_size = mrcfile.open(self.path, header_only=True).voxel_size.x
        self.transform = Transform()
        self.texture = None
        self.quad_va = None
        self.border_va = None
        self.interpolate = False
        self.alpha = 1.0
        self.contrast_lims = [0, 1000]
        self.corner_positions_local = []
        self.set_slice(0, False)
        self.setup_opengl_objects()

    def setup_opengl_objects(self):
        self.texture = Texture(format="r32f")
        self.texture.update(self.data.astype(np.float32))
        self.quad_va = VertexArray()
        self.border_va = VertexArray(attribute_format="xy")
        self.generate_va()
        self.interpolate = not self.interpolate
        self.toggle_interpolation()

    def toggle_interpolation(self):
        self.interpolate = not self.interpolate
        if self.interpolate:
            self.texture.set_linear_mipmap_interpolation()
        else:
            self.texture.set_no_interpolation()

    def generate_va(self):
        # set up the quad vertex array
        w, h = self.width * 0.5, self.height * 0.5
        vertex_attributes = list()
        indices = list()
        n = cfg.ce_va_subdivision
        for i in range(n):
            for j in range(n):
                x = ((2 * i / (n - 1)) - 1) * w
                y = ((2 * j / (n - 1)) - 1) * h
                u = 0.5 + x / w / 2
                v = 0.5 + y / h / 2
                vertex_attributes += [x, y, 0.0, u, v]

        for i in range(n - 1):
            for j in range(n - 1):
                idx = i * n + j
                indices += [idx, idx + 1, idx + n, idx + n, idx + 1, idx + n + 1]

        self.corner_positions_local = [[-w, h], [-w, -h], [w, -h], [w, h]]
        self.quad_va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))

        # set up the border vertex array
        vertex_attributes = [-w, h,
                             w, h,
                             w, -h,
                             -w, -h]
        indices = [0, 1, 1, 2, 2, 3, 3, 0]
        self.border_va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))

    def set_slice(self, requested_slice, update_texture=True):
        if requested_slice == self.current_slice:
            return
        mrc = mrcfile.mmap(self.path, mode="r")
        self.n_slices = mrc.data.shape[0]
        requested_slice = min([max([requested_slice, 0]), self.n_slices - 1])
        self.data = mrc.data[requested_slice, :, :]
        target_type_dict = {np.float32: float, float: float, np.int8: np.uint8, np.int16: np.uint16}
        if type(self.data[0, 0]) not in target_type_dict:
            target_type = float
        else:
            target_type = target_type_dict[type(self.data[0, 0])]
        self.data = np.array(self.data.astype(target_type, copy=False), dtype=float)
        self.current_slice = requested_slice
        for s in self.features:
            s.set_slice(self.current_slice)
        if update_texture:
            self.update_image_texture()

    def update_image_texture(self):
        self.texture.update(self.data.astype(np.float32))

    def update_model_matrix(self):
        self.transform.scale = self.pixel_size
        self.transform.compute_matrix()
        for i in range(4):
            vec = np.matrix([*self.corner_positions_local[i], 0.0, 1.0]).T
            corner_pos = (self.transform.matrix * vec)[0:2]
            self.corner_positions_local[i] = [float(corner_pos[0]), float(corner_pos[1])]

    def world_to_pixel_coordinate(self, world_coordinate):
        vec = np.matrix([world_coordinate[0], world_coordinate[1], 0.0, 1.0]).T
        invmat = np.linalg.inv(self.transform.matrix)
        out_vec = invmat * vec
        pixel_coordinate = [int(out_vec[0, 0] + self.width / 2), int(out_vec[1, 0] + self.height / 2)]
        return pixel_coordinate

    @staticmethod
    def from_clemframe(clemframe):
        se_frame = SEFrame(clemframe.path)

    def __eq__(self, other):
        if isinstance(other, SEFrame):
            return self.uid == other.uid
        return False


class Segmentation:
    idgen = count(0)

    def __init__(self, parent_frame, title):
        uid_counter = next(Segmentation.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        self.parent = parent_frame
        self.parent.active_feature = self
        self.width = self.parent.width
        self.height = self.parent.height
        self.title = title
        self.colour = np.random.randint(0, 4, 3) / 3
        self.colour /= np.sum(self.colour**2)*0.5
        self.alpha = 1.0
        self.brushsize = 10.0
        self.slices = dict()
        self.slices[self.parent.current_slice] = np.zeros((self.width, self.height), dtype=float)
        self.data = self.slices[self.parent.current_slice]
        self.texture = Texture(format="r32f")
        self.texture.update(self.data)

    def set_slice(self, requested_slice):
        if requested_slice in self.slices:
            self.data = self.slices[requested_slice]
            self.texture.update(self.data)
        else:
            self.slices[requested_slice] = None


class Brush:
    # circular_roi = np.zeros(1)
    # circular_roi_radius = -1

    # @staticmethod
    # def set_circular_roi_radius(radius):
    #     if Brush.circular_roi_radius == radius:
    #         return
    #     Brush.circular_roi_radius = radius
    #     Brush.circular_roi = np.zeros((2*radius+1, 2*radius+1))
    #     r = radius**2
    #     for x in range(0, 2*radius+1):
    #         for y in range(0, 2*radius+1):
    #             if ((x-radius)**2 + (y-radius)**2) < r:
    #                 Brush.circular_roi[x, y] = 1

    @staticmethod
    def apply_circular(segmentation, center_coordinates, value=1):
        r = int(segmentation.brushsize)
        dtype = type(segmentation.data[0, 0])
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                x = center_coordinates[0] + i
                y = center_coordinates[1] + j
                if (i**2 + j**2) > r**2:
                    continue
                if 0 <= x < segmentation.width and 0 <= y < segmentation.height:
                    print(f"Painting at: {x}, {y}; exp. dimensions {segmentation.width}, {segmentation.height}; act. dims {segmentation.data.shape}")
                    segmentation.data[x, y] = value
                    segmentation.texture.set_pixel(x, y, dtype(value))


class Renderer:

    def __init__(self):
        self.quad_shader = Shader(os.path.join(cfg.root, "shaders", "se_quad_shader.glsl"))
        self.segmentation_shader = Shader(os.path.join(cfg.root, "shaders", "se_segmentation_shader.glsl"))
        self.border_shader = Shader(os.path.join(cfg.root, "shaders", "se_border_shader.glsl"))

    def render_frame(self, se_frame, camera):
        se_frame.update_model_matrix()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)

        # render the image (later: to framebuffer)
        self.quad_shader.bind()
        se_frame.quad_va.bind()
        se_frame.texture.bind(0)
        self.quad_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.quad_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        self.quad_shader.uniform1f("alpha", se_frame.alpha)
        self.quad_shader.uniform3f("contrastMin", [se_frame.contrast_lims[0], 0.0, 0.0])
        self.quad_shader.uniform3f("contrastMax", [se_frame.contrast_lims[1], 0.0, 0.0])
        glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.quad_shader.unbind()
        se_frame.quad_va.unbind()
        glActiveTexture(GL_TEXTURE0)

        # filter framebuffer?

        # render overlays
        self.segmentation_shader.bind()
        se_frame.quad_va.bind()
        self.segmentation_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.segmentation_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        for segmentation in se_frame.features:
            if segmentation == se_frame.active_feature:
                continue
            self.segmentation_shader.uniform1f("alpha", segmentation.alpha)
            self.segmentation_shader.uniform3f("colour", segmentation.colour)
            segmentation.texture.bind(0)
            glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        # draw active feature last
        segmentation = se_frame.active_feature
        self.segmentation_shader.uniform1f("alpha", segmentation.alpha)
        self.segmentation_shader.uniform3f("colour", segmentation.colour)
        segmentation.texture.bind(0)
        glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        se_frame.quad_va.unbind()
        self.segmentation_shader.unbind()
        glActiveTexture(GL_TEXTURE0)

        # render border
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR)
        glBlendEquation(GL_FUNC_ADD)
        self.border_shader.bind()
        se_frame.border_va.bind()
        self.border_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.border_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        glDrawElements(GL_LINES, se_frame.border_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.border_shader.unbind()
        se_frame.border_va.unbind()


class Camera:
    def __init__(self):
        self.view_matrix = np.identity(4)
        self.projection_matrix = np.identity(4)
        self.view_projection_matrix = np.identity(4)
        self.position = np.zeros(3)
        self.zoom = 1.0
        self.projection_width = 1
        self.projection_height = 1
        self.set_projection_matrix(cfg.window_width, cfg.window_height)

    def cursor_to_world_position(self, cursor_pos):
        """
        cursor_pus: list, [x, y]
        returns: [world_pos_x, world_pos_y]
        Converts an input cursor position to corresponding world position. Assuming orthographic projection matrix.
        """
        inverse_matrix = np.linalg.inv(self.view_projection_matrix)
        window_coordinates = (2 * cursor_pos[0] / cfg.window_width - 1, 1 - 2 * cursor_pos[1] / cfg.window_height)
        window_vec = np.matrix([*window_coordinates, 1.0, 1.0]).T
        world_vec = (inverse_matrix * window_vec)
        return [float(world_vec[0]), float(world_vec[1])]

    def world_to_screen_position(self, world_position):
        vec = np.matrix([world_position[0], world_position[1], 0.0, 1.0]).T
        vec_out = self.view_projection_matrix * vec
        screen_x = int((1 + float(vec_out[0])) * self.projection_width / 2.0)
        screen_y = int((1 - float(vec_out[1])) * self.projection_height / 2.0)
        return [screen_x, screen_y]

    def set_projection_matrix(self, window_width, window_height):
        self.projection_matrix = np.matrix([
            [2 / window_width, 0, 0, 0],
            [0, 2 / window_height, 0, 0],
            [0, 0, -2 / 100, 0],
            [0, 0, 0, 1],
        ])
        self.projection_width = window_width
        self.projection_height = window_height

    def on_update(self):
        self.view_matrix = np.matrix([
            [self.zoom, 0.0, 0.0, self.position[0] * self.zoom],
            [0.0, self.zoom, 0.0, self.position[1] * self.zoom],
            [0.0, 0.0, 1.0, self.position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.view_projection_matrix = np.matmul(self.projection_matrix, self.view_matrix)


class Transform:
    def __init__(self):
        self.translation = np.array([0.0, 0.0])
        self.rotation = 0.0
        self.scale = 1.0
        self.matrix = np.identity(4)
        self.matrix_no_scale = np.identity(4)

    def compute_matrix(self):
        scale_mat = np.identity(4) * self.scale
        scale_mat[3, 3] = 1

        rotation_mat = np.identity(4)
        _cos = np.cos(self.rotation / 180.0 * np.pi)
        _sin = np.sin(self.rotation / 180.0 * np.pi)
        rotation_mat[0, 0] = _cos
        rotation_mat[1, 0] = _sin
        rotation_mat[0, 1] = -_sin
        rotation_mat[1, 1] = _cos

        translation_mat = np.identity(4)
        translation_mat[0, 3] = self.translation[0]
        translation_mat[1, 3] = self.translation[1]

        self.matrix = np.matmul(translation_mat, np.matmul(rotation_mat, scale_mat))
        self.matrix_no_scale = np.matmul(translation_mat, rotation_mat)

    def __add__(self, other):
        out = Transform()
        out.translation[0] = self.translation[0] + other.translation[0]
        out.translation[1] = self.translation[1] + other.translation[1]
        out.rotation = self.rotation + other.rotation
        return out

    def __str__(self):
        return f"Transform with translation = {self.translation[0], self.translation[1]}, scale = {self.scale}, rotation = {self.rotation}"

    def __sub__(self, other):
        out = Transform()
        out.translation[0] = self.translation[0] - other.translation[0]
        out.translation[1] = self.translation[1] - other.translation[1]
        out.rotation = self.rotation - other.rotation
        return out