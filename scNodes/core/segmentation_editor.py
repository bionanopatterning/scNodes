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


        sef = SEFrame("C:/Users/mgflast/Desktop/tomo4.mrc")
        cfg.se_frames.append(sef)
        cfg.se_active_frame = sef

        sef.features.append(Segmentation(sef, "def", (0.0, 0.0, 1.0, 1.0)))
        sef.features.append(Segmentation(sef, "def", (1.0, 0.0, 1.0, 1.0)))

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

        imgui.render()
        self.imgui_implementation.render(imgui.get_draw_data())

    def gui_main(self):
        def brush_window():
            imgui.begin("Brush")
            imgui.text("TODO")  # TODO
            # 1) make brush selection window
            # 2) make brush Object
            # 3) make Brush do something to a Segmentation's data and upload that edit to GPU

            imgui.end()
        # render the active frame
        if cfg.se_active_frame is not None:
            self.renderer.render_frame(cfg.se_active_frame, self.camera)

        # imgui windows
        brush_window()

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

        self.n_slices = 0
        self.data = None
        self.features = list()

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

    @staticmethod
    def from_clemframe(clemframe):
        se_frame = SEFrame(clemframe.path)

    def __eq__(self, other):
        if isinstance(other, SEFrame):
            return self.uid == other.uid
        return False


class Segmentation:
    idgen = count(0)

    def __init__(self, parent_frame, title, colour):
        uid_counter = next(Segmentation.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        self.parent = parent_frame
        self.width = self.parent.width
        self.height = self.parent.height
        self.title = title
        self.colour = colour
        self.alpha = 1.0
        self.data = np.zeros((self.height, self.width))
        self.texture = Texture(format="r32f")
        self.texture.update(self.data)



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