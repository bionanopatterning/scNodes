import glfw
import imgui
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
from opengl_classes import *
import settings
import time
from itertools import count
import config as cfg
from copy import copy
from PIL import Image


class CorrelationEditor:
    if True:
        COLOUR_WINDOW_BACKGROUND = (0.94, 0.94, 0.94, 1.0)
        COLOUR_PANEL_BACKGROUND = (0.94, 0.94, 0.94, 1.0)
        COLOUR_TITLE_BACKGROUND = (0.87, 0.87, 0.83, 0.96)
        COLOUR_FRAME_BACKGROUND = (0.87, 0.87, 0.83, 0.96)
        COLOUR_FRAME_ACTIVE = (0.91, 0.91, 0.86, 1.0)
        COLOUR_FRAME_DARK = (0.83, 0.83, 0.76, 1.0)
        COLOUR_FRAME_EXTRA_DARK = (0.76, 0.76, 0.71, 1.0)
        COLOUR_MAIN_MENU_BAR = (0.882, 0.882, 0.882, 1.0)
        COLOUR_MAIN_MENU_BAR_TEXT = (0.0, 0.0, 0.0, 1.0)
        COLOUR_MAIN_MENU_BAR_HILIGHT = (0.96, 0.95, 0.92, 1.0)
        COLOUR_MENU_WINDOW_BACKGROUND = (0.96, 0.96, 0.96, 1.0)
        COLOUR_TEXT = (0.0, 0.0, 0.0, 1.0)
        COLOUR_DROP_TARGET = (0.83, 0.83, 0.83, 1.0)
        WINDOW_ROUNDING = 5.0

        COLOUR_IMAGE_BORDER = (0.0, 0.0, 0.0, 1.0)
        THICKNESS_IMAGE_BORDER = GLfloat(3.0)

        BLEND_MODES = dict()  # blend mode template: ((glBlendFunc, ARG1, ARG2), (glBlendEquation, ARG1))
        BLEND_MODES[" Alpha blending"] = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_FUNC_ADD)
        BLEND_MODES[" Sum frames"] = (GL_SRC_ALPHA, GL_DST_ALPHA, GL_FUNC_ADD)  ## or maybe GL_ONE instead of GL_DST_ALPHA
        BLEND_MODES[" Subtract frames"] = (GL_SRC_ALPHA, GL_DST_ALPHA, GL_FUNC_SUBTRACT)
        BLEND_MODES[" Subtract frames (inverted)"] = (GL_SRC_ALPHA, GL_DST_ALPHA, GL_FUNC_REVERSE_SUBTRACT)
        BLEND_MODES[" Retain minimum"] = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_MIN)
        BLEND_MODES[" Retain maximum"] = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_MAX)
        BLEND_MODES_LIST = list(BLEND_MODES.keys())

        TRANSLATION_INCREMENT = 0.0
        TRANSLATION_INCREMENT_LARGE = 0.0
        ROTATION_INCREMENT = 0.0
        ROTATION_INCREMENT_LARGE = 0.0
        PIXEL_SIZE_INCREMENT = 0.0
        PIXEL_SIZE_INCREMENT_LARGE = 0.0

        WORLD_PIXEL_SIZE = 100.0
        CAMERA_ZOOM_STEP = 0.1

        HISTOGRAM_BINS = 40

        INFO_PANEL_WIDTH = 200
        INFO_HISTOGRAM_HEIGHT = 70
        INFO_LUT_PREVIEW_HEIGHT = 10

        TOOLTIP_APPEAR_DELAY = 1.0
        TOOLTIP_HOVERED_TIMER = 0.0
        TOOLTIP_HOVERED_START_TIME = 0.0

        ALPHA_SLIDER_WIDTH = 450
        ALPHA_SLIDER_H_OFFSET = 0
        ALPHA_SLIDER_ROUNDING = 50.0
        BLEND_COMBO_WIDTH = 150
        VISUALS_CTRL_ALPHA = 0.8

    # editing
    MOUSE_SHORT_PRESS_MAX_DURATION = 0.25  # seconds
    mouse_left_press_world_pos = [0, 0]
    mouse_left_release_world_pos = [0, 0]

    # data
    frames = list()  # order of frames in this list determines the rendering order. Index 0 = front, index -1 = back.
    gizmos = list()
    gizmo_mode_scale = True  # if False, gizmo mode is rotate instead.
    active_frame = None
    active_frame_timer = 0.0
    active_frame_original_translation = [0, 0]
    active_gizmo = None

    frame_drag_payload = None

    context_menu_open = None
    context_menu_position = [0, 0]
    context_menu_obj = None

    def __init__(self, window, shared_font_atlas=None):
        # Note that CorrelationEditor and NodeEditor share a window. In either's on_update, end_frame,
        # __init__, etc. methods, bits of the same window-management related code are called.

        self.window = window
        self.window.clear_color = CorrelationEditor.COLOUR_WINDOW_BACKGROUND
        #self.window.set_force_alpha_zero(True)
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
        EditorGizmo.init_textures()
        self.gizmos.append(EditorGizmo(EditorGizmo.TYPE_SCALE, idx=0))
        self.gizmos.append(EditorGizmo(EditorGizmo.TYPE_SCALE, idx=1))
        self.gizmos.append(EditorGizmo(EditorGizmo.TYPE_SCALE, idx=2))
        self.gizmos.append(EditorGizmo(EditorGizmo.TYPE_SCALE, idx=3))
        self.gizmos.append(EditorGizmo(EditorGizmo.TYPE_ROTATE, idx=0))
        self.gizmos.append(EditorGizmo(EditorGizmo.TYPE_ROTATE, idx=1))
        self.gizmos.append(EditorGizmo(EditorGizmo.TYPE_ROTATE, idx=2))
        self.gizmos.append(EditorGizmo(EditorGizmo.TYPE_ROTATE, idx=3))
        self.gizmos.append(EditorGizmo(EditorGizmo.TYPE_PIVOT))

        ## DEBUG
        CorrelationEditor.frames.append(CLEMFrame(np.asarray(Image.open("ce_test_refl.tif"))))
        CorrelationEditor.frames.append(CLEMFrame(np.asarray(Image.open("ce_test_fluo.tif"))))
        self.active_frame = CorrelationEditor.frames[0]

    def on_update(self):
        imgui.set_current_context(self.imgui_context)
        self.window.make_current()
        self.window.set_full_viewport()
        if self.window.focused:
            self.imgui_implementation.process_inputs()
        self.window.on_update()
        imgui.new_frame()

        # update all the transformation matrices - for frames as well as for gizmos. Also set which gizmos are visible.
        for frame in CorrelationEditor.frames:
            frame.update_model_matrix_full()
        if self.active_frame is not None:
            frame_corner_coordinates = self.active_frame.corner_positions_local
            visible_gizmo_types = [EditorGizmo.TYPE_SCALE] if CorrelationEditor.gizmo_mode_scale else [EditorGizmo.TYPE_ROTATE, EditorGizmo.TYPE_PIVOT]
            for gizmo in self.gizmos:
                gizmo.hide = gizmo.type not in visible_gizmo_types
                if gizmo.idx is not None:
                    gizmo.transform.translation = copy(frame_corner_coordinates[gizmo.idx])
                    # gizmo center is now moved to the corner of the active frame IN LOCAL COORDINATES.
                elif gizmo.type == EditorGizmo.TYPE_PIVOT:
                    gizmo.transform.translation = copy(self.active_frame.pivot_point)
                    # or in case of type pivot, it is moved to the frame's pivot position.
                # next, update the full matrix:
                gizmo.update_model_matrix(self.active_frame.transform)
        else:
            for gizmo in self.gizmos:
                gizmo.hide = True
        CorrelationEditor.active_frame_timer += self.window.delta_time

        ## content
        self.user_input()
        self.camera_control()
        self.camera.on_update()
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
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *CorrelationEditor.COLOUR_FRAME_DARK)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *CorrelationEditor.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB, *CorrelationEditor.COLOUR_FRAME_DARK)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB_HOVERED, *CorrelationEditor.COLOUR_FRAME_DARK)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_GRAB_ACTIVE, *CorrelationEditor.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SCROLLBAR_BACKGROUND, *CorrelationEditor.COLOUR_FRAME_EXTRA_DARK)
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *CorrelationEditor.COLOUR_TEXT)
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, *CorrelationEditor.COLOUR_TEXT)
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *CorrelationEditor.COLOUR_FRAME_EXTRA_DARK)
        imgui.push_style_color(imgui.COLOR_MENUBAR_BACKGROUND, *CorrelationEditor.COLOUR_MAIN_MENU_BAR)
        imgui.push_style_color(imgui.COLOR_DRAG_DROP_TARGET, *(1.0, 0.0, 1.0, 1.0))  ## TODO change
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, CorrelationEditor.WINDOW_ROUNDING)

        for frame in reversed(CorrelationEditor.frames):
            self.renderer.render_frame_quad(self.camera, frame)
        if self.active_frame is not None:
            self.renderer.render_frame_border(self.camera, self.active_frame)
        for gizmo in self.gizmos:
            self.renderer.render_gizmo(self.camera, gizmo)
        self.menu_bar()
        self.context_menu()
        self.frame_info_window()
        self.objects_info_window()
        self.visuals_window()

        imgui.pop_style_color(25)
        imgui.pop_style_var(1)

    def menu_bar(self):
        imgui.push_style_color(imgui.COLOR_MENUBAR_BACKGROUND, *CorrelationEditor.COLOUR_MAIN_MENU_BAR)
        imgui.push_style_color(imgui.COLOR_TEXT, *CorrelationEditor.COLOUR_MAIN_MENU_BAR_TEXT)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *CorrelationEditor.COLOUR_MAIN_MENU_BAR_HILIGHT)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *CorrelationEditor.COLOUR_MAIN_MENU_BAR_HILIGHT)
        imgui.push_style_color(imgui.COLOR_HEADER, *CorrelationEditor.COLOUR_MAIN_MENU_BAR_HILIGHT)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *CorrelationEditor.COLOUR_MENU_WINDOW_BACKGROUND)

        if imgui.core.begin_main_menu_bar():
            if imgui.begin_menu("File"):
                pass # TODO
                imgui.end_menu()
            if imgui.begin_menu("Settings"):
                _c, self.window.clear_color = imgui.color_edit4("Background colour", *self.window.clear_color, flags=imgui.COLOR_EDIT_NO_INPUTS)
                imgui.end_menu()
            if imgui.begin_menu("Editor"):
                select_node_editor, _ = imgui.menu_item("Node Editor", None, False)
                select_correlation_editor, _ = imgui.menu_item("Correlation", None, True)
                if select_node_editor:
                    cfg.active_editor = 0
                imgui.end_menu()
            imgui.end_main_menu_bar()
        imgui.pop_style_color(6)

    def frame_info_window(self):
        af = self.active_frame
        imgui.push_style_color(imgui.COLOR_HEADER, *CorrelationEditor.COLOUR_PANEL_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *CorrelationEditor.COLOUR_PANEL_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *CorrelationEditor.COLOUR_PANEL_BACKGROUND)
        imgui.set_next_window_size_constraints((CorrelationEditor.INFO_PANEL_WIDTH, -1), (CorrelationEditor.INFO_PANEL_WIDTH, -1))
        imgui.begin("Selected frame parameters", False, imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE)
        #  Transform info
        expanded, _ = imgui.collapsing_header("Transform", None)
        if expanded and af is not None:
            _t = af.transform
            imgui.push_item_width(70)
            _, _t.translation[0] = imgui.input_float("x", _t.translation[0], CorrelationEditor.TRANSLATION_INCREMENT, CorrelationEditor.TRANSLATION_INCREMENT_LARGE, '%.1f nm', imgui.INPUT_TEXT_AUTO_SELECT_ALL)
            imgui.same_line(spacing=20)
            _, _t.translation[1] = imgui.input_float("y", _t.translation[1], CorrelationEditor.TRANSLATION_INCREMENT, CorrelationEditor.TRANSLATION_INCREMENT_LARGE, '%.1f nm', imgui.INPUT_TEXT_AUTO_SELECT_ALL)
            _, _t.rotation = imgui.input_float("rotation", _t.rotation, CorrelationEditor.ROTATION_INCREMENT, CorrelationEditor.ROTATION_INCREMENT_LARGE, '%.2f°', imgui.INPUT_TEXT_AUTO_SELECT_ALL)
            _, af.pixel_size = imgui.input_float("pixel size (nm)", af.pixel_size, CorrelationEditor.PIXEL_SIZE_INCREMENT, CorrelationEditor.PIXEL_SIZE_INCREMENT_LARGE, '%.1f', imgui.INPUT_TEXT_AUTO_SELECT_ALL)

        # Visuals info
        expanded, _ = imgui.collapsing_header("Visuals", None)
        if expanded and af is not None:
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

        imgui.end()
        imgui.pop_style_color(3)


    def objects_info_window(self):
        imgui.push_style_color(imgui.COLOR_TEXT_SELECTED_BACKGROUND, *CorrelationEditor.COLOUR_FRAME_BACKGROUND)

        def render_frame_as_node(f, recursive=False):
            if f.parent is None or recursive:
                tree_node_flags = imgui.TREE_NODE_DEFAULT_OPEN | imgui.TREE_NODE_BULLET
                if self.active_frame == f:
                    tree_node_flags = tree_node_flags | imgui.TREE_NODE_SELECTED
                if imgui.tree_node(f.title + f"##{f.uid}_trnd", tree_node_flags):
                    if imgui.is_item_clicked():
                        self.active_frame = f
                    if imgui.begin_drag_drop_source(imgui.DRAG_DROP_SOURCE_NO_PREVIEW_TOOLTIP):
                        CorrelationEditor.frame_drag_payload = f
                        imgui.set_drag_drop_payload("none", b'0')  # actual handling of payload doesn't work in pyimgui 2.0.1
                        imgui.end_drag_drop_source()
                    if imgui.begin_drag_drop_target():
                        if self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE):
                            CorrelationEditor.frame_drag_payload.parent_to(f)
                            CorrelationEditor.frame_drag_payload = None
                        imgui.end_drag_drop_target()

                    for child in f.children:
                        render_frame_as_node(child, recursive=True)
                    imgui.tree_pop()

        if imgui.begin("Frames in scene", False, imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE):
            for frame in CorrelationEditor.frames:
                render_frame_as_node(frame)
            # if the drop payload still exists _AND_ user released mouse outside of drop target, unparent the payload
            if CorrelationEditor.frame_drag_payload and self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE):
                CorrelationEditor.frame_drag_payload.unparent()
            imgui.end()
        imgui.pop_style_color(1)

    def visuals_window(self):
        imgui.begin("##visualsw", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_BACKGROUND | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, CorrelationEditor.ALPHA_SLIDER_ROUNDING)
        imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, CorrelationEditor.ALPHA_SLIDER_ROUNDING)
        imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1.0)

        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *CorrelationEditor.COLOUR_PANEL_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *CorrelationEditor.COLOUR_PANEL_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *CorrelationEditor.COLOUR_PANEL_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *CorrelationEditor.COLOUR_TITLE_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *CorrelationEditor.COLOUR_TITLE_BACKGROUND)

        if self.active_frame is not None:
            cw = imgui.get_content_region_available_width()

            imgui.set_next_item_width(CorrelationEditor.ALPHA_SLIDER_WIDTH)
            _, self.active_frame.alpha = imgui.slider_float("##alpha", self.active_frame.alpha, 0.0, 1.0, format="alpha = %.2f")

            # blend mode
            imgui.same_line()
            imgui.set_next_item_width(CorrelationEditor.BLEND_COMBO_WIDTH)
            _, self.active_frame.blend_mode = imgui.combo("##blending", self.active_frame.blend_mode, CorrelationEditor.BLEND_MODES_LIST)


        imgui.pop_style_color(5)
        imgui.pop_style_var(3)
        imgui.end()

    def camera_control(self):
        if imgui.get_io().want_capture_mouse:
            return None
        if self.window.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE):
            delta_cursor = self.window.cursor_delta
            self.camera.position[0] += delta_cursor[0]
            self.camera.position[1] -= delta_cursor[1]
        if self.window.get_key(glfw.KEY_LEFT_SHIFT):
            self.camera.zoom *= (1.0 + self.window.scroll_delta[1] * CorrelationEditor.CAMERA_ZOOM_STEP)

    def user_input(self):
        if imgui.get_io().want_capture_mouse:
            return None
        # If left mouse click, find which (if any) object was clicked and set it to active.
        if self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.PRESS):
            # check which object should be active (if any)
            clicked_object = self.get_object_under_cursor(self.window.cursor_pos)
            print(clicked_object)
            if isinstance(clicked_object, CLEMFrame):
                if self.active_frame != clicked_object:
                    CorrelationEditor.gizmo_mode_scale = True
                    CorrelationEditor.active_frame_timer = 0.0
                self.active_frame = clicked_object
                self.active_gizmo = None
            elif isinstance(clicked_object, EditorGizmo):
                self.active_gizmo = clicked_object
            else:
                self.active_gizmo = None
                self.active_frame = None
            self.mouse_left_press_world_pos = self.camera.cursor_to_world_position(self.window.cursor_pos)
        elif self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, max_duration=min([CorrelationEditor.active_frame_timer, CorrelationEditor.MOUSE_SHORT_PRESS_MAX_DURATION])):  # TODO: this elif shouldn't fire _right_ after the PRESS event above - it causes the gizmo to go from scale to rotate right upon activating a new frame
            clicked_object = self.get_object_under_cursor(self.window.cursor_pos)
            if self.active_frame == clicked_object:
                CorrelationEditor.gizmo_mode_scale = not CorrelationEditor.gizmo_mode_scale

        if self.active_gizmo is None and self.active_frame is not None:
            if self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT):  # user is dragging the active frame
                cursor_world_pos_now = self.camera.cursor_to_world_position(self.window.cursor_pos)
                cursor_world_pos_prev = self.camera.cursor_to_world_position(self.window.cursor_pos_previous_frame)
                delta_x = cursor_world_pos_now[0] - cursor_world_pos_prev[0]
                delta_y = cursor_world_pos_now[1] - cursor_world_pos_prev[1]
                self.active_frame.transform.translation[0] += delta_x
                self.active_frame.transform.translation[1] += delta_y

        elif self.active_frame is None:
            pass  # TODO - behaviour dependent on which gizmo is active.
        else:
            pass  # No gizmo or frame active, so no mouse input (yet).

        if self.window.get_mouse_event(glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS):
            print("Right click")
            self.context_menu_obj = self.get_object_under_cursor(self.window.cursor_pos)
            self.context_menu_position = copy(self.window.cursor_pos)
            self.context_menu_open = True

    def context_menu(self):
        if self.context_menu_open:
            if isinstance(self.context_menu_obj, CLEMFrame):
                frame = self.context_menu_obj
                imgui.set_next_window_position(self.context_menu_position[0] - 3, self.context_menu_position[1] - 3)
                imgui.begin("##ce_cm", False, imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SAVED_SETTINGS)
                ## context menu options
                any_option_selected = True
                if imgui.menu_item("Send to top")[0]:
                    frame.move_to_front()
                elif imgui.menu_item("Send to bottom")[0]:
                    frame.move_to_back()
                elif imgui.menu_item("Raise one step")[0]:
                    frame.move_forwards()
                elif imgui.menu_item("Lower one step")[0]:
                    frame.move_backwards()
                else:
                    any_option_selected = False
                if self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT) and not imgui.is_window_hovered(imgui.HOVERED_CHILD_WINDOWS | imgui.HOVERED_ALLOW_WHEN_BLOCKED_BY_POPUP):
                    self.context_menu_open = False

                imgui.end()

    def end_frame(self):
        self.window.end_frame()

    def get_object_under_cursor(self, cursor_position):
        """In this function, the cursor position is first translated into the corresponding world position, and
        subsequently it is checked whether that world position lies within the quad of any of the existing CLEMFrame
        objects. """
        def is_point_in_rectangle(point, corner_positions):
            def triangle_area(a, b, c):
                area = abs((b[0] * a[1] - a[0] * b[1]) +
                           (c[0] * b[1] - b[0] * c[1]) +
                           (a[0] * c[1] - c[0] * a[1])
                           ) / 2
                return area

            P = point
            A = corner_positions[0]
            B = corner_positions[1]
            C = corner_positions[2]
            D = corner_positions[3]
            point_edge_triangles_total_area = (triangle_area(P, A, D) +
                                               triangle_area(P, D, C) +
                                               triangle_area(P, C, B) +
                                               triangle_area(P, B, A))
            rectangle_area = np.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2) * np.sqrt((A[0] - D[0])**2 + (A[1] - D[1])**2)
            return point_edge_triangles_total_area <= rectangle_area

        cursor_world_position = self.camera.cursor_to_world_position(cursor_position)
        for gizmo in CorrelationEditor.gizmos:
            if gizmo.hide:
                continue
            if is_point_in_rectangle(cursor_world_position, gizmo.corner_positions):
                return gizmo
        for frame in CorrelationEditor.frames:
            if frame.hide:
                continue
            if is_point_in_rectangle(cursor_world_position, frame.corner_positions):
                return frame
        return None

    def get_camera_zoom(self):
        return self.camera.zoom

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
        self.gizmo_shader = Shader("shaders/ce_gizmo_shader.glsl")

    def render_frame_quad(self, camera, frame):
        blend_mode = CorrelationEditor.BLEND_MODES[CorrelationEditor.BLEND_MODES_LIST[frame.blend_mode]]
        glBlendFunc(blend_mode[0], blend_mode[1])
        glBlendEquation(blend_mode[2])
        if not frame.hide:
            self.quad_shader.bind()
            frame.quad_va.bind()
            frame.texture.bind(0)
            frame.lut_texture.bind(1)
            self.quad_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
            self.quad_shader.uniform2f("contrastLimits", frame.contrast_lims)
            self.quad_shader.uniform1f("alpha", frame.alpha)
            self.quad_shader.uniformmat4("modelMatrix", frame.transform_full.matrix)
            glDrawElements(GL_TRIANGLES, frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            self.quad_shader.unbind()
            frame.quad_va.unbind()
            glActiveTexture(GL_TEXTURE0)

    def render_frame_border(self, camera, frame):
        pass  # set blend mode to something that accentuates the border regardless of colour below. #TODO
        #glEnable(GL_BLEND)
        #glBlendEquation(GL_FUNC_REVERSE_SUBTRACT)
        if not frame.hide:
            self.border_shader.bind()
            frame.border_va.bind()
            self.border_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
            self.border_shader.uniformmat4("modelMatrix", frame.transform_full.matrix)
            self.border_shader.uniform3f("lineColour", CorrelationEditor.COLOUR_IMAGE_BORDER)
            glDrawElements(GL_LINES, frame.border_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            self.border_shader.unbind()
            frame.border_va.unbind()

    def render_gizmo(self, camera, gizmo):
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR)
        glBlendEquation(GL_FUNC_ADD)
        if not gizmo.hide:
            self.gizmo_shader.bind()
            gizmo.va.bind()
            gizmo.texture.bind(0)
            self.gizmo_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
            self.gizmo_shader.uniformmat4("modelMatrix", gizmo.transform_full.matrix)
            glDrawElements(GL_TRIANGLES, gizmo.va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            self.gizmo_shader.unbind()
            gizmo.va.unbind()
            glActiveTexture(GL_TEXTURE0)


class Camera:
    def __init__(self):
        self.view_matrix = np.identity(4)
        self.projection_matrix = np.identity(4)
        self.view_projection_matrix = np.identity(4)
        self.position = np.zeros(3)
        self.zoom = 1.0
        self.set_projection_matrix(settings.ne_window_width, settings.ne_window_height)

    def cursor_to_world_position(self, cursor_pos):
        """Converts an input cursor position to corresponding world position. Assuming orthographic projection matrix."""
        inverse_matrix = np.linalg.inv(self.view_projection_matrix)
        window_coordinates = (2 * cursor_pos[0] / settings.ne_window_width - 1, 1 - 2 * cursor_pos[1] / settings.ne_window_height)
        window_vec = np.matrix([*window_coordinates, 1.0, 1.0]).T
        world_vec = (inverse_matrix * window_vec)
        return [float(world_vec[0]), float(world_vec[1])]

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
        self.height, self.width = self.data.shape
        self.children = list()
        self.parent = None
        self.tree_level = 0
        self.title = "Frame "+str(self.uid)
        # transform parameters
        self.pixel_size = 100.0  # pixel size in nm
        self.pivot_point = np.zeros(2)  # pivot point for rotation and scaling of this particular image. can be moved by the user. In _local coordinates_, i.e. relative to where the frame itself is positioned.
        self.transform = Transform()
        self.transform_full = Transform()
        self.flip = False  # 1.0 or -1.0, in case the image is mirrored.

        # visuals
        self.blend_mode = 0
        self.lut = 1
        self.colour = (1.0, 1.0, 1.0, 1.0)
        self.alpha = 1.0
        self.contrast_lims = [0.0, 65535.0]
        self.compute_autocontrast()
        self.hist_bins = list()
        self.hist_vals = list()
        self.compute_histogram()
        self.hide = False

        # aux
        self.corner_positions_local = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.corner_positions = [[0, 0], [0, 0], [0, 0], [0, 0]]

        # opengl
        self.texture = Texture(format="r32f")
        self.texture.update(self.data.astype(np.float32))
        self.lut_texture = Texture(format="rgb32f")
        self.quad_va = VertexArray()
        self.vertex_positions = list()
        self.border_va = VertexArray(attribute_format="xy")
        self.update_lut()
        self.generate_va()

    def update_model_matrix_full(self):
        self.transform.scale = self.pixel_size / CorrelationEditor.WORLD_PIXEL_SIZE
        self.transform.compute_matrix()

        self.transform_full = copy(self.transform)
        _parent = self.parent
        while _parent is not None:
            self.transform_full += _parent.transform
            _parent = _parent.parent
        self.transform_full.compute_matrix()

        # update corner positions
        for i in range(4):
            local_corner_pos = tuple(self.corner_positions_local[i])
            vec = np.matrix([*local_corner_pos, 0.0, 1.0]).T
            world_corner_pos = self.transform_full.matrix * vec
            self.corner_positions[i] = [float(world_corner_pos[0]), float(world_corner_pos[1])]

    def unparent(self):
        if self.parent is not None:
            self.parent.children.remove(self)
        self.tree_level = 0
        self.parent = None

    def parent_to(self, parent):
        if self == parent:
            return

        # check is parent isn't a child of self
        if parent.is_in_tree_of(self):
            return
        # remove self from parent's children list
        self.unparent()
        self.tree_level = parent.tree_level + 1
        self.parent = parent
        parent.children.append(self)

    def is_in_tree_of(self, parent):
        pass ## TODO check per level in the faimlty tree

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
        self.corner_positions_local = [[-w, h], [-w, -h], [w, -h], [w, h]]
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
        print(CorrelationEditor.frames)

    def move_to_back(self):
        CorrelationEditor.frames.append(CorrelationEditor.frames.pop(CorrelationEditor.frames.index(self)))
        print(CorrelationEditor.frames)

    def move_backwards(self):
        idx = CorrelationEditor.frames.index(self)
        if idx < (len(CorrelationEditor.frames) - 1):
            CorrelationEditor.frames[idx], CorrelationEditor.frames[idx + 1] = CorrelationEditor.frames[idx + 1], CorrelationEditor.frames[idx]
        print(CorrelationEditor.frames)

    def move_forwards(self):
        idx = CorrelationEditor.frames.index(self)
        if idx > 0:
            CorrelationEditor.frames[idx], CorrelationEditor.frames[idx - 1] = CorrelationEditor.frames[idx - 1], CorrelationEditor.frames[idx]
        print(CorrelationEditor.frames)

    def __eq__(self, other):
        if isinstance(other, CLEMFrame):
            return self.uid == other.uid
        return False

    def __str__(self):
        return f"CLEMFrame with id {self.uid}."


class EditorGizmo:
    idgen = count(0)
    HITBOX_SIZE = 7
    TYPE_SCALE = 0
    TYPE_ROTATE = 1
    TYPE_PIVOT = 2

    ICON_TEXTURES = dict()

    @staticmethod
    def init_textures():
        icon_scale = np.asarray(Image.open("icons/icon_scale_256.png")).astype(np.uint8)
        icon_rotate = np.asarray(Image.open("icons/icon_rotate_256.png")).astype(np.uint8)
        icon_pivot = np.asarray(Image.open("icons/icon_pivot_256.png")).astype(np.uint8)
        icon_scale[:, :, 0:2] = 1
        icon_rotate[:, :, 0:2] = 1
        icon_pivot[:, :, 0:2] = 1
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_SCALE] = Texture(format="rgba8u")
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_SCALE].update(icon_scale)
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_ROTATE] = Texture(format="rgba8u")
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_ROTATE].update(icon_rotate)
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_PIVOT] = Texture(format="rgba8u")
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_PIVOT].update(icon_pivot)

    def __init__(self, gizmo_type=TYPE_SCALE, idx=None):
        self.uid = next(EditorGizmo.idgen)
        self.type = gizmo_type
        self.texture = EditorGizmo.ICON_TEXTURES[self.type]
        self.va = VertexArray()
        self.corner_positions_local = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.corner_positions = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.idx = idx
        self.generate_va()
        self.hide = False
        self.transform = Transform()
        self.transform_full = Transform()
        self.translation_offset = [0, 0]
        if self.idx is not None:
            _s = EditorGizmo.HITBOX_SIZE + (2.0 if self.type == EditorGizmo.TYPE_SCALE else 0.0)
            self.translation_offset[0] = -_s * np.cos((self.idx - 0.5) * np.pi / 2.0)
            self.translation_offset[1] = -_s * np.sin((self.idx - 0.5) * np.pi / 2.0)
            self.transform.rotation = self.idx * 90.0

    def generate_va(self):
        w, h = EditorGizmo.HITBOX_SIZE, EditorGizmo.HITBOX_SIZE
        vertex_attributes = [-w, h, 1.0, 0.0, 1.0,
                             -w, -h, 1.0, 0.0, 0.0,
                             w, -h, 1.0, 1.0, 0.0,
                             w, h, 1.0, 1.0, 1.0]
        self.corner_positions_local = [[-w, h], [-w, -h], [w, -h], [w, h]]  #TODO: these values should be updated every frame in proportion to camera zoom (in order to make gizmo size independent of camera zoom)
        indices = [0, 1, 2, 2, 0, 3]
        self.va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))

    def update_model_matrix(self, parent_transform):
        self.transform.translation[0] += self.translation_offset[0]
        self.transform.translation[1] += self.translation_offset[1]
        self.transform_full = copy(self.transform) + parent_transform
        self.transform.compute_matrix()
        self.transform_full.compute_matrix()
        self.transform.translation[0] -= self.translation_offset[0]
        self.transform.translation[1] -= self.translation_offset[1]
        # update hitbox corner positions
        for i in range(4):
            local_corner_pos = tuple(self.corner_positions_local[i])
            vec = np.matrix([*local_corner_pos, 0.0, 1.0]).T
            world_corner_pos = self.transform_full.matrix * vec
            self.corner_positions[i] = [float(world_corner_pos[0]), float(world_corner_pos[1])]


class Transform:
    def __init__(self):
        self.translation = np.array([0.0, 0.0])
        self.rotation = 0.0
        self.scale = 1.0
        self.matrix = np.identity(4)

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

    def __add__(self, other):
        out = Transform()
        out.translation = self.translation + other.translation
        out.rotation = self.rotation + other.rotation
        out.scale = self.scale * other.scale  # check if product works intuitively, else change to sum
        return out

    def __str__(self):
        return f"Transform with translation = {self.translation[0], self.translation[1]}, scale = {self.scale}, rotation = {self.rotation}"



