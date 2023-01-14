import importlib
import glfw
import imgui
import numpy as np
from imgui.integrations.glfw import GlfwRenderer
import util
from opengl_classes import *
import settings
import time
from itertools import count
import config as cfg
from copy import copy, deepcopy
from PIL import Image
import mrcfile
from tkinter import filedialog
import pyperclip
import tifffile
from dataset import Frame
import glob
#from clemframe import CLEMFrame, Transform
from ceplugin import *

class CorrelationEditor:
    if True:
        FRAME_LIST_INDENT_WIDTH = 20.0
        COLOUR_IMAGE_BORDER = (1.0, 1.0, 1.0, 0.0)
        THICKNESS_IMAGE_BORDER = GLfloat(3.0)

        BLEND_MODES = dict()  # blend mode template: ((glBlendFunc, ARG1, ARG2), (glBlendEquation, ARG1))
        BLEND_MODES[" Transparency"] = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_FUNC_ADD)
        BLEND_MODES[" Sum"] = (GL_SRC_ALPHA, GL_DST_ALPHA, GL_FUNC_ADD)
        BLEND_MODES[" Subtract"] = (GL_SRC_ALPHA, GL_DST_ALPHA, GL_FUNC_REVERSE_SUBTRACT)
        BLEND_MODES[" Subtract (inverted)"] = (GL_SRC_ALPHA, GL_DST_ALPHA, GL_FUNC_SUBTRACT)
        BLEND_MODES[" Retain minimum"] = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_MIN)
        BLEND_MODES[" Retain maximum"] = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_MAX)
        BLEND_MODES[" Multiply"] = (GL_DST_COLOR, GL_ONE_MINUS_SRC_ALPHA, GL_FUNC_ADD)
        BLEND_MODES_LIST = list(BLEND_MODES.keys())

        TRANSLATION_SPEED = 1.0
        ROTATION_SPEED = 1.0
        SCALE_SPEED = 1.0

        SNAPPING_DISTANCE = 1000.0
        SNAPPING_ACTIVATION_DISTANCE = 5000.0
        DEFAULT_IMAGE_PIXEL_SIZE = 100.0
        DEFAULT_HORIZONTAL_FOV_WIDTH = 50000  # upon init, camera zoom is such that from left to right of window = 50 micron.
        DEFAULT_ZOOM = 1.0  # adjusted in init
        DEFAULT_WORLD_PIXEL_SIZE = 1.0  # adjusted on init
        CAMERA_ZOOM_STEP = 0.1

        HISTOGRAM_BINS = 40

        INFO_PANEL_WIDTH = 200
        INFO_HISTOGRAM_HEIGHT = 70
        INFO_LUT_PREVIEW_HEIGHT = 10

        TOOLTIP_APPEAR_DELAY = 1.0
        TOOLTIP_HOVERED_TIMER = 0.0
        TOOLTIP_HOVERED_START_TIME = 0.0

        FRAMES_IN_SCENE_WINDOW_WIDTH = 300

        ALPHA_SLIDER_WIDTH = 450
        ALPHA_SLIDER_H_OFFSET = 0
        ALPHA_SLIDER_ROUNDING = 50.0
        BLEND_COMBO_WIDTH = 150
        VISUALS_CTRL_ALPHA = 0.8

        CAMERA_MAX_ZOOM = 0.7

        SCALE_BAR_HEIGHT = 6.0
        SCALE_BAR_WINDOW_MARGIN = 5.0
        SCALE_BAR_COLOUR = (0.0, 0.0, 0.0, 1.0)
        scale_bar_size = 5000.0
        show_scale_bar = False
        scale_bar_world_position = [0, 0]

        measure_active = False
        measure_tool = None
        measure_tool_colour = (255 / 255, 202 / 255, 28 / 255, 1.0)
        # editing
        MOUSE_SHORT_PRESS_MAX_DURATION = 0.25  # seconds
        ARROW_KEY_TRANSLATION = 100.0  # nm
        ARROW_KEY_TRANSLATION_FAST = 1000.0  # nm
        ARROW_KEY_TRANSLATION_SLOW = 10.0
        mouse_left_press_world_pos = [0, 0]
        mouse_left_release_world_pos = [0, 0]

        mrc_flip_on_load = False
        flip_images_on_load = True

        # data
        frames = list()  # order of frames in this list determines the rendering order. Index 0 = front, index -1 = back.
        gizmos = list()
        gizmo_mode_scale = True  # if False, gizmo mode is rotate instead.
        active_frame = None
        active_frame_timer = 0.0
        active_frame_original_translation = [0, 0]
        active_gizmo = None
        transform_buffer = None
        snap_enabled = True

        frame_drag_payload = None
        release_frame_drag_payload = False
        context_menu_open = None
        context_menu_position = [0, 0]
        context_menu_obj = None

        incoming_frame_buffer = list()
        frames_dropped = False

        renderer = None


        # export
        export_roi_mode = 1  # 0 for ROI, 1 for Image
        export_roi_obj = None
        export_roi_render = False
        ex_lims = [-10000, -10000, 10000, 10000]
        ex_pxnm = 10.0
        ex_img_size = [0, 0]
        ex_path = ""
        ex_png = True
        EXPORT_FBO = None
        EXPORT_TILE_SIZE = 1000
        EXPORT_SIZE_CHANGE_SPEED = 0.05
        EXPORT_RESOLUTION_CHANGE_SPEED = 0.05
        EXPORT_ROI_LINE_COLOUR = (255 / 255, 202 / 255, 28 / 255, 1.0)

        # tools
        tool_factory = dict()
        tools_list = list()
        tool_descriptions = list()
        current_tool = None
        selected_tool = 0
        location_gizmo = None
        location_gizmo_visible = False

    def __init__(self, window, imgui_context, imgui_impl):
        self.window = window
        self.window.clear_color = cfg.COLOUR_WINDOW_BACKGROUND
        self.window.make_current()

        self.imgui_context = imgui_context
        self.imgui_implementation = imgui_impl

        CorrelationEditor.renderer = Renderer()
        self.camera = Camera()  # default camera 'zoom' value is 1.0, in which case the vertical field of view size is equal to window_height_in_pixels nanometer.
        CorrelationEditor.DEFAULT_ZOOM = cfg.window_height / CorrelationEditor.DEFAULT_HORIZONTAL_FOV_WIDTH  # now it is DEFAULT_HORIZONTAL_FOV_WIDTH
        CorrelationEditor.DEFAULT_WORLD_PIXEL_SIZE = 1.0 / CorrelationEditor.DEFAULT_ZOOM
        self.camera.zoom = CorrelationEditor.DEFAULT_ZOOM

        CorrelationEditor.measure_tool = MeasureTool()
        CorrelationEditor.export_roi_obj = ExportROI()

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
        CorrelationEditor.location_gizmo = EditorGizmo(EditorGizmo.TYPE_LOCATION)
        self.gizmos.append(CorrelationEditor.location_gizmo)

        CorrelationEditor.init_toolkit()
        # Export FBO
        CorrelationEditor.EXPORT_FBO = FrameBuffer(CorrelationEditor.EXPORT_TILE_SIZE, CorrelationEditor.EXPORT_TILE_SIZE, texture_format="rgba32f")
        # load icons
        if True:
            self.icon_no_interp = Texture(format="rgba32f")
            pxd_icon_no_interp = np.asarray(Image.open("icons/icon_ninterp_256.png")).astype(np.float32) / 255.0
            self.icon_no_interp.update(pxd_icon_no_interp)
            self.icon_no_interp.set_linear_interpolation()  # :)
            self.icon_linterp = Texture(format="rgba32f")
            pxd_icon_linterp = np.asarray(Image.open("icons/icon_linterp_256.png")).astype(np.float32) / 255.0
            self.icon_linterp.update(pxd_icon_linterp)
            self.icon_linterp.set_linear_interpolation()
            self.icon_close = Texture(format="rgba32f")
            pxd_icon_close = np.asarray(Image.open("icons/icon_close_256.png")).astype(np.float32) / 255.0
            self.icon_close.update(pxd_icon_close)
            self.icon_close.set_linear_interpolation()

    def on_update(self):
        imgui.set_current_context(self.imgui_context)
        self.window.make_current()
        self.window.set_full_viewport()
        if self.window.focused:
            self.imgui_implementation.process_inputs()

        if imgui.is_key_pressed(glfw.KEY_GRAVE_ACCENT):
            cfg.active_editor = 0

        if cfg.correlation_editor_relink:
            CorrelationEditor.relink_after_load()
            cfg.correlation_editor_relink = False

        incoming_files = deepcopy(self.window.dropped_files)
        self.window.on_update()
        if self.window.window_size_changed:
            cfg.window_width = self.window.width
            cfg.window_height = self.window.height
            self.camera.set_projection_matrix(cfg.window_width, cfg.window_height)
        imgui.get_io().display_size = self.window.width, self.window.height
        imgui.new_frame()

        # Handle frames sent to CE from external source
        if incoming_files != list():
            self.load_externally_dropped_files(incoming_files)

        # GUI
        self.editor_control()
        self.camera_control()
        self.camera.on_update()
        self.gui_main()

        for frame in cfg.ce_frames:
            frame.update_model_matrix()

        # Handle scale, rotate, and pivot gizmos:
        if CorrelationEditor.active_frame is not None:
            visible_gizmo_types = [EditorGizmo.TYPE_SCALE] if CorrelationEditor.gizmo_mode_scale else [EditorGizmo.TYPE_ROTATE, EditorGizmo.TYPE_PIVOT]
            for gizmo in self.gizmos:
                gizmo.hide = gizmo.gizmo_type not in visible_gizmo_types
                if not gizmo.hide:
                    gizmo.set_parent_frame(CorrelationEditor.active_frame, self.camera.zoom)
                    gizmo.set_zoom_compensation_factor(self.camera.zoom)
        else:
            for gizmo in self.gizmos:
                    gizmo.hide = True
        # Handle location indicator gizmo:
        if CorrelationEditor.location_gizmo_visible:
            CorrelationEditor.location_gizmo.hide = False
            CorrelationEditor.location_gizmo.set_no_parent_frame(self.camera.zoom)
            CorrelationEditor.location_gizmo.set_zoom_compensation_factor(self.camera.zoom)
        CorrelationEditor.active_frame_timer += self.window.delta_time

        # Handle incoming frames
        CorrelationEditor.frames_dropped = False
        for frame_obj in CorrelationEditor.incoming_frame_buffer:
            CorrelationEditor.frames_dropped = True
            pxd = frame_obj.load()
            new_frame = CLEMFrame(pxd)
            new_frame.pixel_size = frame_obj.pixel_size
            new_frame.lut = frame_obj._ce_lut
            new_frame.contrast_lims = [frame_obj._ce_clims[0], frame_obj._ce_clims[1]]
            new_frame.rgb_contrast_lims = copy(frame_obj._ce_clims)
            new_frame.update_lut()
            cfg.ce_frames.insert(0, new_frame)
            CorrelationEditor.active_frame = new_frame
        CorrelationEditor.incoming_frame_buffer = list()

        ## end content
        imgui.render()
        self.imgui_implementation.render(imgui.get_draw_data())

    def load_externally_dropped_files(self, paths):
        for path in paths:
            incoming = ImportedFrameData(path)
            clem_frame = incoming.to_CLEMFrame()
            if clem_frame:
                cfg.ce_frames.insert(0, clem_frame)
                CorrelationEditor.active_frame = clem_frame

    @staticmethod
    def relink_after_load():
        for frame in cfg.ce_frames:
            frame.setup_opengl_objects()
        CorrelationEditor.active_frame = None

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

        first_frame = True
        for frame in reversed(cfg.ce_frames):
            CorrelationEditor.renderer.render_frame_quad(self.camera, frame, override_blending=False)  # previously: override_blending = first_frame
            first_frame = False
        if CorrelationEditor.active_frame is not None:
            CorrelationEditor.renderer.render_frame_border(self.camera, CorrelationEditor.active_frame)
        for gizmo in self.gizmos:
            CorrelationEditor.renderer.render_gizmo(self.camera, gizmo)
        if CorrelationEditor.measure_tool:
            CorrelationEditor.renderer.render_measure_tool(self.camera, CorrelationEditor.measure_tool)
        if CorrelationEditor.export_roi_render:
            CorrelationEditor.renderer.render_roi(self.camera, CorrelationEditor.export_roi_obj)
        self.menu_bar()
        self.context_menu()
        self.tool_info_window()
        self.objects_info_window()
        self.visuals_window()
        self.measure_tools()
        self._warning_window()

        imgui.pop_style_color(28)
        imgui.pop_style_var(1)

    def menu_bar(self):
        imgui.push_style_color(imgui.COLOR_MENUBAR_BACKGROUND, *cfg.COLOUR_MAIN_MENU_BAR)
        imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_MAIN_MENU_BAR_TEXT)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *cfg.COLOUR_MAIN_MENU_BAR_HILIGHT)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *cfg.COLOUR_MAIN_MENU_BAR_HILIGHT)
        imgui.push_style_color(imgui.COLOR_HEADER, *cfg.COLOUR_MAIN_MENU_BAR_HILIGHT)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *cfg.COLOUR_MENU_WINDOW_BACKGROUND)

        if imgui.core.begin_main_menu_bar():
            if imgui.begin_menu("File"):
                if imgui.menu_item("Save project")[0]:
                    try:
                        filename = filedialog.asksaveasfilename(filetypes=[("srNodes project", ".srnp")])
                        if filename != '':
                            cfg.save_project(filename)
                    except Exception as e:
                        cfg.set_error(e, f"Error saving project?\n")
                if imgui.menu_item("Load project")[0]:
                    try:
                        filename = filedialog.askopenfilename(filetypes=[("srNodes project", ".srnp")])
                        if filename != '':
                            cfg.load_project(filename)
                    except Exception as e:
                        cfg.set_error(e, f"Error loading project - are you sure you selected a '.srnp' file?\n")
                imgui.end_menu()
            if imgui.begin_menu("Settings"):
                _c, self.window.clear_color = imgui.color_edit4("Background colour", *self.window.clear_color, flags=imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_SIDE_PREVIEW)
                cfg.ce_clear_colour = self.window.clear_color
                imgui.set_next_item_width(60)
                _c, CorrelationEditor.ARROW_KEY_TRANSLATION = imgui.input_float("Arrow key step size (nm)", CorrelationEditor.ARROW_KEY_TRANSLATION, 0.0, 0.0, "%.1f")
                _c, CorrelationEditor.snap_enabled = imgui.checkbox("Allow snapping", CorrelationEditor.snap_enabled)
                _c, cfg.ce_flip_mrc_on_load = imgui.checkbox("Flip .mrc volumes when loading", cfg.ce_flip_mrc_on_load)
                _c, cfg.ce_flip_on_load = imgui.checkbox("Flip images when loading", cfg.ce_flip_on_load)
                imgui.end_menu()

            if imgui.begin_menu("Editor"):
                select_node_editor, _ = imgui.menu_item("Node Editor", None, False)
                select_correlation_editor, _ = imgui.menu_item("Correlation", None, True)
                if select_node_editor:
                    cfg.active_editor = 0
                imgui.end_menu()
            imgui.end_main_menu_bar()
        imgui.pop_style_color(6)

    def tool_info_window(self):
        af = CorrelationEditor.active_frame
        imgui.push_style_color(imgui.COLOR_HEADER, *cfg.COLOUR_PANEL_BACKGROUND[0:3], 0.0)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *cfg.COLOUR_PANEL_BACKGROUND[0:3], 0.0)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *cfg.COLOUR_PANEL_BACKGROUND[0:3], 0.0)
        imgui.set_next_window_size_constraints((CorrelationEditor.INFO_PANEL_WIDTH, -1), (CorrelationEditor.INFO_PANEL_WIDTH, -1))
        imgui.set_next_window_position(0, 18, imgui.ONCE)
        imgui.begin("Tools", False, imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_TITLE_BAR)

        #  Transform info
        expanded, _ = imgui.collapsing_header("Transform", None)
        if expanded and af is not None:
            _t = af.transform
            imgui.push_item_width(90)
            imgui.text("         X: ")
            imgui.same_line()
            dx = _t.translation[0]
            _, _t.translation[0] = imgui.drag_float("##X", _t.translation[0], CorrelationEditor.TRANSLATION_SPEED, 0.0, 0.0, f"{_t.translation[0]:.1f} nm")
            dx -= _t.translation[0]
            imgui.text("         Y: ")
            imgui.same_line()
            dy = _t.translation[1]
            _, _t.translation[1] = imgui.drag_float("##Y", _t.translation[1], CorrelationEditor.TRANSLATION_SPEED, 0.0, 0.0, f"{_t.translation[1]:.1f} nm")
            dy -= _t.translation[1]
            imgui.text("     Angle: ")
            imgui.same_line()
            _, _t.rotation = imgui.drag_float("##Angle", _t.rotation, CorrelationEditor.ROTATION_SPEED, 0.0, 0.0, '%.2f°')
            imgui.text("Pixel size: ")
            imgui.same_line()
            _, af.pixel_size = imgui.drag_float("##Pixel size", af.pixel_size, CorrelationEditor.SCALE_SPEED, 0.0, 0.0, '%.3f nm')
            af.pixel_size = max([af.pixel_size, 0.1])
            af.pivot_point[0] += dx
            af.pivot_point[1] += dy
            imgui.spacing()

        # Visuals info
        expanded, _ = imgui.collapsing_header("Visuals", None)
        if expanded and af is not None:
            if af.is_rgb:  # rgb histograms
                _cw = imgui.get_content_region_available_width()
                imgui.push_item_width(_cw)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0.0, 0.0))
                imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 5.0)
                # Red histogram
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *(0.8, 0.1, 0.1, 1.0))
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *(0.8, 0.1, 0.1, 1.0))
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *(0.9, 0.1, 0.1, 1.0))
                imgui.plot_histogram("##histR", af.rgb_hist_vals[0],
                                     graph_size=(_cw, CorrelationEditor.INFO_HISTOGRAM_HEIGHT // 2))
                _c, af.rgb_contrast_lims[0] = imgui.slider_float("#minr", af.rgb_contrast_lims[0], 0, 255,
                                                                 format='min %.0f')
                _c, af.rgb_contrast_lims[1] = imgui.slider_float("#maxr", af.rgb_contrast_lims[1], 0, 255,
                                                                 format='max %.0f')
                imgui.pop_style_color(3)

                # Green histogram
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *(0.1, 0.8, 0.1, 1.0))
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *(0.1, 0.8, 0.1, 1.0))
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *(0.1, 0.9, 0.1, 1.0))
                imgui.plot_histogram("##histG", af.rgb_hist_vals[1],
                                     graph_size=(_cw, CorrelationEditor.INFO_HISTOGRAM_HEIGHT // 2))
                _c, af.rgb_contrast_lims[2] = imgui.slider_float("#ming", af.rgb_contrast_lims[2], 0, 255,
                                                                 format='min %.0f')
                _c, af.rgb_contrast_lims[3] = imgui.slider_float("#maxg", af.rgb_contrast_lims[3], 0, 255,
                                                                 format='max %.0f')
                imgui.pop_style_color(3)

                # Blue histogram
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *(0.1, 0.1, 0.8, 1.0))
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *(0.1, 0.1, 0.9, 1.0))
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *(0.1, 0.1, 0.8, 1.0))
                imgui.plot_histogram("##histB", af.rgb_hist_vals[2],
                                     graph_size=(_cw, CorrelationEditor.INFO_HISTOGRAM_HEIGHT // 2))
                _c, af.rgb_contrast_lims[4] = imgui.slider_float("#minb", af.rgb_contrast_lims[4], 0, 255,
                                                                 format='min %.0f')
                _c, af.rgb_contrast_lims[5] = imgui.slider_float("#maxb", af.rgb_contrast_lims[5], 0, 255,
                                                                 format='max %.0f')
                imgui.pop_style_color(3)

                imgui.pop_item_width()
                imgui.pop_style_var(2)
            else:
                # LUT
                imgui.text("Look-up table")
                imgui.set_next_item_width(129 + (27 if af.lut != 0 else 0.0))
                _clut, af.lut = imgui.combo("##LUT", af.lut, ["Custom colour"] + settings.lut_names, len(settings.lut_names) + 1)
                if af.lut == 0:
                    imgui.same_line()
                    _c, af.colour = imgui.color_edit4("##lutclr", *af.colour, imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                    _clut = _clut or _c
                if _clut:
                    af.update_lut()
                imgui.same_line(spacing=-1)
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
                _uv_left = 0.5
                _uv_right = 0.5
                if _max != _min:
                    _uv_left = 1.0 + (_l - _max) / (_max - _min)
                    _uv_right = 1.0 + (_h - _max) / (_max - _min)
                imgui.image(af.lut_texture.renderer_id, _cw - 1, CorrelationEditor.INFO_LUT_PREVIEW_HEIGHT, (_uv_left, 0.5), (_uv_right, 0.5), border_color=cfg.COLOUR_FRAME_BACKGROUND)
                imgui.push_item_width(_cw)
                _c, af.contrast_lims[0] = imgui.slider_float("min", af.contrast_lims[0], af.hist_bins[0], af.hist_bins[-1], format='min %.1f')
                _c, af.contrast_lims[1] = imgui.slider_float("max", af.contrast_lims[1], af.hist_bins[0], af.hist_bins[-1], format='max %.1f')
                imgui.pop_item_width()
                imgui.text("Saturation:")
                imgui.same_line()
                _cw = imgui.get_content_region_available_width()
                imgui.set_next_item_width(_cw)
                _c, af.lut_clamp_mode = imgui.combo("##clamping", af.lut_clamp_mode, ["Clamp", "Discard", "Discard min"])
                if _c:
                    af.update_lut()
                CorrelationEditor.tooltip("Set whether saturated pixels (with intensities outside of the min/max range)\n"
                                          "are clamped to the min/max value, or discarded (alpha set to 0.0)")
                imgui.spacing()

        CorrelationEditor.export_roi_render = False
        if imgui.collapsing_header("Export", None)[0]:
            CorrelationEditor.export_roi_render = True
            # Export mode selection buttons
            _cw = imgui.get_content_region_available_width()
            _button_width = (_cw - 10) / 2
            _button_height = 20
            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, cfg.CE_WIDGET_ROUNDING)
            if CorrelationEditor.export_roi_mode == 0:
                imgui.push_style_color(imgui.COLOR_BUTTON, *cfg.COLOUR_FRAME_DARK)
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, *cfg.COLOUR_FRAME_BACKGROUND)
            if imgui.button("ROI", _button_width, _button_height): CorrelationEditor.export_roi_mode = 0
            imgui.pop_style_color()
            if CorrelationEditor.export_roi_mode == 1:
                imgui.push_style_color(imgui.COLOR_BUTTON, *cfg.COLOUR_FRAME_DARK)
            else:
                imgui.push_style_color(imgui.COLOR_BUTTON, *cfg.COLOUR_FRAME_BACKGROUND)
            imgui.same_line()
            if imgui.button("Selection", _button_width, _button_height): CorrelationEditor.export_roi_mode = 1
            imgui.pop_style_color()
            imgui.pop_style_var(1)
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (2.0, 1.0))
            # X/Y limit widget
            imgui.spacing()
            imgui.push_item_width(50)
            lims = np.asarray(CorrelationEditor.ex_lims) / 1000.0
            imgui.text(" X:")
            imgui.same_line()
            _a, lims[0] = imgui.drag_float("##export_xmin", lims[0], CorrelationEditor.EXPORT_SIZE_CHANGE_SPEED, format = f'{lims[0]:.2f}')
            imgui.same_line()
            imgui.text("to")
            imgui.same_line()
            _b, lims[2] = imgui.drag_float("##export_xmax", lims[2], CorrelationEditor.EXPORT_SIZE_CHANGE_SPEED, format = f'{lims[2]:.2f}')
            imgui.same_line()
            imgui.text("um")
            imgui.text(" Y:")
            imgui.same_line()
            _c, lims[1] = imgui.drag_float("##export_ymin", lims[1], CorrelationEditor.EXPORT_SIZE_CHANGE_SPEED, format = f'{lims[1]:.2f}')
            imgui.same_line()
            imgui.text("to")
            imgui.same_line()
            _d, lims[3] = imgui.drag_float("##export_ymax", lims[3], CorrelationEditor.EXPORT_SIZE_CHANGE_SPEED, format = f'{lims[3]:.2f}')
            if _a or _b or _c or _d:
                CorrelationEditor.export_roi_mode = 0
            imgui.same_line()
            imgui.text("um")
            CorrelationEditor.ex_lims = lims * 1000.0
            CorrelationEditor.export_roi_obj.set_roi(CorrelationEditor.ex_lims)
            imgui.spacing()
            # Pixels per nm + width/height
            imgui.text("Resolution:")
            imgui.same_line()
            imgui.push_item_width(40)
            _c, CorrelationEditor.ex_pxnm = imgui.drag_float("##export_pxnm", CorrelationEditor.ex_pxnm, CorrelationEditor.EXPORT_RESOLUTION_CHANGE_SPEED, format = f'%.1f')
            imgui.pop_style_var()
            CorrelationEditor.ex_pxnm = max([0.001, CorrelationEditor.ex_pxnm])
            CorrelationEditor.tooltip(f"equals {CorrelationEditor.ex_pxnm * 25400000:.0f} dpi :)")
            imgui.same_line()
            imgui.text("nm / px")
            if CorrelationEditor.export_roi_mode == 1:  # export by selection
                af = CorrelationEditor.active_frame
                if af is not None:
                    corners = np.asarray(af.corner_positions)
                    left = np.amin(corners[:, 0])
                    right = np.amax(corners[:, 0])
                    top = np.amin(corners[:, 1])
                    bottom = np.amax(corners[:, 1])
                    CorrelationEditor.ex_lims = [left, top, right, bottom]
            else:
                imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0.0, 0.0))
                imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)
                imgui.push_style_var(imgui.STYLE_WINDOW_MIN_SIZE, (0.0, 0.0))
                imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *CorrelationEditor.EXPORT_ROI_LINE_COLOUR)
                CorrelationEditor.EXPORT_ROI_HANDLE_SIZE = 7
                force_square = False
                # Top left marker
                roi = CorrelationEditor.ex_lims
                s = CorrelationEditor.EXPORT_ROI_HANDLE_SIZE
                screen_pos = self.camera.world_to_screen_position([roi[0], roi[3]])
                imgui.set_next_window_position(screen_pos[0] - s // 2, screen_pos[1] - s // 2)
                imgui.set_next_window_size(s, s)
                imgui.begin("##roi_top_left", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)
                if imgui.is_mouse_down(glfw.MOUSE_BUTTON_LEFT) and imgui.is_window_focused():
                    cursor_world_pos_prev = self.camera.cursor_to_world_position(self.window.cursor_pos_previous_frame)
                    cursor_world_pos_current = self.camera.cursor_to_world_position(self.window.cursor_pos)
                    delta_x = cursor_world_pos_current[0] - cursor_world_pos_prev[0]
                    delta_y = cursor_world_pos_current[1] - cursor_world_pos_prev[1]
                    CorrelationEditor.ex_lims[0] += delta_x
                    CorrelationEditor.ex_lims[3] += delta_y
                    if imgui.is_key_down(glfw.KEY_LEFT_SHIFT):
                        force_square = True
                imgui.end()

                # Bottom right marker
                screen_pos = self.camera.world_to_screen_position([roi[2], roi[1]])
                imgui.set_next_window_position(screen_pos[0] - s // 2, screen_pos[1] - s // 2)
                imgui.set_next_window_size(s, s)
                imgui.begin("##roi_bottom_right", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)
                if imgui.is_mouse_down(glfw.MOUSE_BUTTON_LEFT) and imgui.is_window_focused():
                    cursor_world_pos_prev = self.camera.cursor_to_world_position(self.window.cursor_pos_previous_frame)
                    cursor_world_pos_current = self.camera.cursor_to_world_position(self.window.cursor_pos)
                    delta_x = cursor_world_pos_current[0] - cursor_world_pos_prev[0]
                    delta_y = cursor_world_pos_current[1] - cursor_world_pos_prev[1]
                    CorrelationEditor.ex_lims[2] += delta_x
                    CorrelationEditor.ex_lims[1] += delta_y
                    if imgui.is_key_down(glfw.KEY_LEFT_SHIFT):
                        force_square = True
                imgui.end()

                if force_square:
                    ex_width = CorrelationEditor.ex_lims[2] - CorrelationEditor.ex_lims[0]
                    ex_height = CorrelationEditor.ex_lims[3] - CorrelationEditor.ex_lims[1]
                    print(ex_width, ex_height)
                    if ex_width > ex_height:
                        delta = ex_width - ex_height
                        CorrelationEditor.ex_lims[2] -= delta / 2
                        CorrelationEditor.ex_lims[0] += delta / 2
                    else:
                        delta = ex_height - ex_width
                        CorrelationEditor.ex_lims[3] -= delta / 2
                        CorrelationEditor.ex_lims[1] += delta / 2

                imgui.pop_style_color()
                imgui.pop_style_var(3)
            CorrelationEditor.ex_img_size = [int(1000.0 * (lims[2] - lims[0]) / CorrelationEditor.ex_pxnm), int(1000.0 * (lims[3] - lims[1]) / CorrelationEditor.ex_pxnm)]
            imgui.text(f"Output size: {CorrelationEditor.ex_img_size[0]} x {CorrelationEditor.ex_img_size[1]}")
            imgui.text("Export as:")
            imgui.text(" ")
            imgui.same_line()
            if imgui.radio_button(".png", CorrelationEditor.ex_png):
                CorrelationEditor.ex_png = not CorrelationEditor.ex_png
            imgui.same_line(spacing = 20)
            if imgui.radio_button(".tif stack", not CorrelationEditor.ex_png):
                CorrelationEditor.ex_png = not CorrelationEditor.ex_png
            imgui.text("Filename:")
            imgui.set_next_item_width(150)
            _c, CorrelationEditor.ex_path = imgui.input_text("##outpath", CorrelationEditor.ex_path, 256)
            CorrelationEditor.tooltip(CorrelationEditor.ex_path)
            imgui.same_line()
            if imgui.button("...", 26, 19):
                selected_file = filedialog.asksaveasfilename()
                if selected_file is not None:
                    CorrelationEditor.ex_path = selected_file
            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, CorrelationEditor.ALPHA_SLIDER_ROUNDING)
            imgui.new_line()
            imgui.same_line(spacing=(_cw - _button_width) / 2.0)
            if imgui.button("Export##button", _button_width, _button_height):
                self.export_image()
            imgui.pop_style_var()
            imgui.spacing()

        if imgui.collapsing_header("Measure", None)[0]:
            imgui.text("Active widgets:")
            imgui.new_line()
            imgui.same_line(spacing=5.0)
            _c, CorrelationEditor.show_scale_bar = imgui.checkbox("Scalebar"+(":" if CorrelationEditor.show_scale_bar else ""), CorrelationEditor.show_scale_bar)
            if CorrelationEditor.show_scale_bar:
                imgui.set_next_item_width(80)
                imgui.same_line()
                _c, CorrelationEditor.scale_bar_size = imgui.drag_float("##scalebarlength", CorrelationEditor.scale_bar_size, 1.0, 0.0, 0.0, format='%.0f nm')
                if imgui.begin_popup_context_item():
                    _c, CorrelationEditor.SCALE_BAR_HEIGHT = imgui.drag_int("##Scale bar thickness", CorrelationEditor.SCALE_BAR_HEIGHT, 1.0, 1.0, 0.0, format = 'Drag me to change scale bar thickness: %i px')
                    CorrelationEditor.SCALE_BAR_HEIGHT = max([1, CorrelationEditor.SCALE_BAR_HEIGHT])
                    imgui.end_popup()
            imgui.new_line()
            imgui.same_line(spacing=5.0)
            _c, CorrelationEditor.measure_active = imgui.checkbox("Measure tool", CorrelationEditor.measure_active)
            if _c:
                CorrelationEditor.measure_tool.reset()
            if CorrelationEditor.measure_active:
                imgui.same_line()
                _c, CorrelationEditor.measure_tool_colour = imgui.color_edit4("##text colour", *CorrelationEditor.measure_tool_colour, flags=imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
        # draw the measure image
        if CorrelationEditor.measure_active and CorrelationEditor.measure_tool.p_set:
            if CorrelationEditor.measure_tool.q_set:
                window_pos = self.camera.world_to_screen_position(CorrelationEditor.measure_tool.q)
            else:
                window_pos = self.window.cursor_pos
            imgui.set_next_window_position(window_pos[0] + 10.0, window_pos[1] - 10.0)
            imgui.begin("##measure_t_outp", False, imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_BACKGROUND)
            distance = CorrelationEditor.measure_tool.distance
            imgui.push_style_color(imgui.COLOR_TEXT, *CorrelationEditor.measure_tool_colour)
            if distance < 100:
                imgui.text(f"{distance:.1f} nm")
            elif distance < 5000:
                imgui.text(f"{distance:.0f} nm")
            else:
                imgui.text(f"{distance / 1000.0:.1f} um")
            imgui.pop_style_color(1)
            imgui.end()

        if imgui.collapsing_header("Plugins", None)[0]:
            cfg.ce_active_frame = CorrelationEditor.active_frame
            _cw = imgui.get_content_region_available_width()
            imgui.set_next_item_width(_cw - 25)
            _c, CorrelationEditor.selected_tool = imgui.combo("##tools", CorrelationEditor.selected_tool, ["Select plugin..."] + CorrelationEditor.tools_list)
            if _c:
                if CorrelationEditor.selected_tool > 0:
                    CorrelationEditor.current_tool = CorrelationEditor.tool_factory[CorrelationEditor.tools_list[CorrelationEditor.selected_tool - 1]]()
                else:
                    CorrelationEditor.current_tool = CEPlugin()
            if CorrelationEditor.selected_tool > 0:
                imgui.same_line()
                imgui.button("?", 19, 19)
                CorrelationEditor.tooltip(CorrelationEditor.tool_descriptions[CorrelationEditor.selected_tool - 1])
            if CorrelationEditor.current_tool is not None:
                imgui.separator()
                try:
                    cfg.ce_selected_position = CorrelationEditor.location_gizmo.transform.translation
                    CorrelationEditor.current_tool.on_update()
                    CorrelationEditor.current_tool.render()
                    CorrelationEditor.location_gizmo_visible = CorrelationEditor.current_tool.FLAG_SHOW_LOCATION_PICKER
                except Exception as e:
                    cfg.set_error(e, f"Plugin '{CorrelationEditor.tools_list[CorrelationEditor.selected_tool - 1]}' caused an error:")

        imgui.end()
        imgui.pop_style_color(3)

    def export_image(self):
        def export_png():
            # Set up
            camera = Camera()
            camera.set_projection_matrix(CorrelationEditor.EXPORT_TILE_SIZE, CorrelationEditor.EXPORT_TILE_SIZE)
            # Compute renderer tile size in nm
            tile_size_pixels = CorrelationEditor.EXPORT_TILE_SIZE
            T = tile_size_pixels
            pixel_size = CorrelationEditor.ex_pxnm
            tile_size = tile_size_pixels * pixel_size
            # Compute required number of tiles
            out_size_pixels = [int(np.ceil((CorrelationEditor.ex_lims[2] - CorrelationEditor.ex_lims[0]) / pixel_size)), int(np.ceil((CorrelationEditor.ex_lims[3] - CorrelationEditor.ex_lims[1]) / pixel_size))]
            W = out_size_pixels[0]
            H = out_size_pixels[1]
            tiles_h = int(np.ceil((CorrelationEditor.ex_lims[2] - CorrelationEditor.ex_lims[0]) / tile_size))
            tiles_v = int(np.ceil((CorrelationEditor.ex_lims[3] - CorrelationEditor.ex_lims[1]) / tile_size))
            camera.zoom = CorrelationEditor.DEFAULT_WORLD_PIXEL_SIZE / pixel_size * CorrelationEditor.DEFAULT_ZOOM
            camera_start_position = [CorrelationEditor.ex_lims[0], CorrelationEditor.ex_lims[3]]
            out_img = np.zeros((out_size_pixels[0], out_size_pixels[1], 4))
            alpha_mask = np.zeros((out_size_pixels[0], out_size_pixels[1]))
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_ALWAYS)
            import matplotlib.pyplot as plt
            for i in range(tiles_h):
                for j in range(tiles_v):
                    offset_x = (0.5 + i) * tile_size
                    offset_y = (0.5 + j) * tile_size
                    camera.position[0] = -(camera_start_position[0] + offset_x)
                    camera.position[1] = -(camera_start_position[1] - offset_y)
                    camera.on_update()
                    CorrelationEditor.EXPORT_FBO.clear([*self.window.clear_color[0:3], 1.0])
                    CorrelationEditor.EXPORT_FBO.bind()
                    glClearDepth(0.0)
                    glClear(GL_DEPTH_BUFFER_BIT)
                    for frame in reversed(cfg.ce_frames):
                        CorrelationEditor.renderer.render_frame_quad(camera, frame)
                    tile = glReadPixels(0, 0, tile_size_pixels, tile_size_pixels, GL_RGB, GL_FLOAT)
                    out_img[i*T:min([(i+1)*T, W]), j*T:min([(j+1)*T, H]), 0:3] = np.rot90(tile, 1, (1, 0))[:min([T, W-(i*T)]), :min([T, H-(j*T)]), :]
                    depth_tile = glReadPixels(0, 0, tile_size_pixels, tile_size_pixels, GL_DEPTH_COMPONENT, GL_FLOAT)
                    alpha_mask[i*T:min([(i+1)*T, W]), j*T:min([(j+1)*T, H])] = np.rot90(depth_tile, 1, (1, 0))[:min([T, W-(i*T)]), :min([T, H-(j*T)])]
            glDisable(GL_DEPTH_TEST)
            CorrelationEditor.EXPORT_FBO.unbind()
            alpha_mask = 1.0 - (alpha_mask < np.amax(alpha_mask)) * 1.0
            out_img[:, :, 3] = alpha_mask
            out_img = np.rot90(out_img, -1, (1, 0))
            out_img = out_img[::-1, :]
            out_img *= 255
            out_img[out_img < 0] = 0
            out_img[out_img > 255] = 255
            out_img = out_img.astype(np.uint8)
            util.save_png(out_img, CorrelationEditor.ex_path, alpha=True)

        def export_tiff_stack():
            # Set up
            camera = Camera()
            camera.set_projection_matrix(CorrelationEditor.EXPORT_TILE_SIZE, CorrelationEditor.EXPORT_TILE_SIZE)
            # Compute renderer tile size in nm
            tile_size_pixels = CorrelationEditor.EXPORT_TILE_SIZE
            T = tile_size_pixels
            pixel_size = CorrelationEditor.ex_pxnm
            tile_size = tile_size_pixels * pixel_size
            # Compute required number of tiles
            out_size_pixels = [int(np.ceil((CorrelationEditor.ex_lims[2] - CorrelationEditor.ex_lims[0]) / pixel_size)),
                               int(np.ceil((CorrelationEditor.ex_lims[3] - CorrelationEditor.ex_lims[1]) / pixel_size))]
            W = out_size_pixels[0]
            H = out_size_pixels[1]
            tiles_h = int(np.ceil((CorrelationEditor.ex_lims[2] - CorrelationEditor.ex_lims[0]) / tile_size))
            tiles_v = int(np.ceil((CorrelationEditor.ex_lims[3] - CorrelationEditor.ex_lims[1]) / tile_size))
            camera.zoom = CorrelationEditor.DEFAULT_WORLD_PIXEL_SIZE / pixel_size * CorrelationEditor.DEFAULT_ZOOM
            camera_start_position = [CorrelationEditor.ex_lims[0], CorrelationEditor.ex_lims[3]]
            tiff_path = CorrelationEditor.ex_path
            if not (tiff_path[-5:] == ".tiff" or tiff_path[-4:] == ".tif"):
                tiff_path += ".tif"

            with tifffile.TiffWriter(tiff_path) as tiffw:
                for frame in reversed(cfg.ce_frames):
                    if frame.hide:
                        continue
                    out_img = np.zeros((out_size_pixels[0], out_size_pixels[1]), dtype=np.float32)
                    frame.force_lut_grayscale()
                    for i in range(tiles_h):
                        for j in range(tiles_v):
                            offset_x = (0.5 + i) * tile_size
                            offset_y = (0.5 + j) * tile_size
                            camera.position[0] = camera_start_position[0] + offset_x
                            camera.position[1] = camera_start_position[1] - offset_y
                            camera.on_update()
                            CorrelationEditor.EXPORT_FBO.clear((0.0, 0.0, 0.0, 0.0))
                            CorrelationEditor.EXPORT_FBO.bind()
                            CorrelationEditor.renderer.render_frame_quad(camera, frame, override_blending=True)
                            tile = glReadPixels(0, 0, tile_size_pixels, tile_size_pixels, GL_RED, GL_FLOAT)
                            out_img[i * T:min([(i + 1) * T, W]), j * T:min([(j + 1) * T, H])] = np.rot90(tile, 3, (1, 0))[:min([T, W - (i * T)]), :min([T, H - (j * T)])]
                    out_img = np.rot90(out_img, 3, (0, 1))
                    out_img = out_img[::-1, :]
                    tiffw.write(out_img, description=frame.title, resolution=(1./(1e-7 * pixel_size), 1./(1e-7 * pixel_size), 'CENTIMETER'))
                    frame.update_lut()
            CorrelationEditor.EXPORT_FBO.unbind()

        if CorrelationEditor.ex_path == "":
            return
        if CorrelationEditor.ex_png:
            export_png()
        else:
            export_tiff_stack()

    def objects_info_window(self):
        content_width = 0
        to_delete = None
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0.0, 0.0))
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_WINDOW_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_BUTTON, *cfg.COLOUR_WINDOW_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_TEXT)

        def frame_info_gui(f, indent=0, enable_drag=True, enable_widgets=True):
            nonlocal content_width, to_delete
            if f.parent is None:
                if CorrelationEditor.frame_drag_payload == f and enable_drag:
                    imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT_FADE)
                else:
                    imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT)
            imgui.push_id(f"{f.uid}_finf")
            if indent != 0:
                imgui.indent(CorrelationEditor.FRAME_LIST_INDENT_WIDTH * indent)
            background_color_active_frame = False
            if CorrelationEditor.active_frame == f:
                imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *(0.0, 0.0, 0.0, 1.0))
                background_color_active_frame = True

            # Name
            imgui.bullet_text("")
            imgui.same_line()
            label_width = CorrelationEditor.FRAMES_IN_SCENE_WINDOW_WIDTH - indent * CorrelationEditor.FRAME_LIST_INDENT_WIDTH - 72 - (0 if f.is_rgb else 15)

            _c, selected = imgui.selectable(""+f.title+f"###fuid{f.uid}", selected=CorrelationEditor.active_frame == f, width=label_width)
            if imgui.begin_popup_context_item():
                _c, f.title = imgui.input_text("##fname", f.title, 30)
                if imgui.menu_item("Send to top")[0]:
                    f.move_to_front()
                elif imgui.menu_item("Send to bottom")[0]:
                    f.move_to_back()
                elif imgui.menu_item("Raise one step")[0]:
                    f.move_forwards()
                elif imgui.menu_item("Lower one step")[0]:
                    f.move_backwards()
                elif imgui.menu_item("Rotate +90° (ccw)")[0]:
                    f.transform.rotation += 90.0
                elif imgui.menu_item("Rotate -90° (cw)")[0]:
                    f.transform.rotation -= 90.0
                elif imgui.menu_item("Flip horizontally")[0]:
                    f.flip()
                elif imgui.menu_item("Flip vertically")[1]:
                    f.flip(horizontally=False)
                if imgui.begin_menu("Render binned"):
                    if imgui.menu_item("None", selected=f.binning==1)[0]:
                        f.binning = 1
                    if imgui.menu_item("1.5 x", selected=f.binning==1.5)[0]:
                        f.binning = 1.5
                    elif imgui.menu_item("2 x", selected=f.binning==2)[0]:
                        f.binning = 2
                    elif imgui.menu_item("3 x", selected=f.binning==3)[0]:
                        f.binning = 3
                    elif imgui.menu_item("4 x", selected=f.binning==4)[0]:
                        f.binning = 4
                    elif imgui.menu_item("8 x", selected=f.binning==8)[0]:
                        f.binning = 8
                    imgui.end_menu()
                imgui.end_popup()
            if selected:
                CorrelationEditor.active_frame = f
            if background_color_active_frame:
                imgui.pop_style_color(1)
            # Drag drop
            if enable_drag:
                imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *(1.0, 1.0, 1.0, 0.0))
                imgui.push_style_color(imgui.COLOR_BORDER, *(1.0, 1.0, 1.0, 0.0))
                imgui.push_style_color(imgui.COLOR_BORDER_SHADOW, *(1.0, 1.0, 1.0, 0.0))
                if imgui.begin_drag_drop_source():
                    CorrelationEditor.frame_drag_payload = f
                    imgui.set_drag_drop_payload("None", b'0')
                    frame_info_gui(f, indent=0, enable_drag=False, enable_widgets=False)
                    imgui.end_drag_drop_source()
                imgui.pop_style_color(3)
                if imgui.begin_drag_drop_target():
                    if self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT):
                        payload = imgui.accept_drag_drop_payload("None")  # do not actually accept payload when user releases mouse, this causes a crash in pyimgui 2.0.1
                    if CorrelationEditor.release_frame_drag_payload:
                        CorrelationEditor.frame_drag_payload.parent_to(f)
                        CorrelationEditor.release_frame_drag_payload = False
                        CorrelationEditor.frame_drag_payload = None
                    imgui.end_drag_drop_target()

            if enable_widgets:
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1.0)
                if not f.is_rgb:
                    imgui.same_line(position=content_width - 35.0)
                    _c, f.colour = imgui.color_edit4("##Background colour", *f.colour, flags=imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP | imgui.COLOR_EDIT_DISPLAY_HEX | imgui.COLOR_EDIT_DISPLAY_RGB)
                    if _c:
                        f.lut = 0  # change lut to custom colour
                        f.update_lut()
                # Hide checkbox
                imgui.same_line(position=content_width - 20.0)
                visible = not f.hide
                _c, visible = imgui.checkbox("##hideframe", visible)
                f.hide = not visible
                CorrelationEditor.tooltip("Hide frame")
                imgui.same_line(position = content_width - 5.0)
                # Delete button
                if imgui.image_button(self.icon_close.renderer_id, 13, 13):
                    to_delete = f
                CorrelationEditor.tooltip("Delete frame")
                imgui.pop_style_var(1)
            if indent != 0:
                imgui.unindent(CorrelationEditor.FRAME_LIST_INDENT_WIDTH * indent)

            for child in f.children:
                frame_info_gui(child, indent+1, enable_drag, enable_widgets)
            if f.parent is None:
                imgui.pop_style_color(1)
            imgui.pop_id()

        if not self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT):
            if CorrelationEditor.release_frame_drag_payload:
                CorrelationEditor.frame_drag_payload = None
            if CorrelationEditor.frame_drag_payload is not None:
                CorrelationEditor.release_frame_drag_payload = True
        imgui.set_next_window_position(self.window.width - CorrelationEditor.FRAMES_IN_SCENE_WINDOW_WIDTH, 18)
        imgui.set_next_window_size(CorrelationEditor.FRAMES_IN_SCENE_WINDOW_WIDTH, self.window.height - 18)
        if imgui.begin(" Frames in scene", False, imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS):
            content_width = imgui.get_window_content_region_width()
            for frame in cfg.ce_frames:
                if frame.parent is None:
                    frame_info_gui(frame)
            # if the drop payload still exists _AND_ user released mouse outside of drop target, unparent the payload
            if CorrelationEditor.frame_drag_payload is not None and CorrelationEditor.release_frame_drag_payload:
                CorrelationEditor.frame_drag_payload.unparent()
                CorrelationEditor.release_frame_drag_payload = False
                CorrelationEditor.frame_drag_payload = None
        if to_delete is not None:
            CorrelationEditor.delete_frame(to_delete)
        imgui.end()
        imgui.pop_style_color(3)
        imgui.pop_style_var(1)

    def measure_tools(self):
        if CorrelationEditor.show_scale_bar:
            imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0.0, 0.0))
            imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)
            imgui.push_style_var(imgui.STYLE_WINDOW_MIN_SIZE, (0.0, 0.0))
            imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *self.window.clear_color)
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *self.window.clear_color)
            world_pixel = CorrelationEditor.DEFAULT_WORLD_PIXEL_SIZE / self.camera.zoom * CorrelationEditor.DEFAULT_ZOOM
            scale_bar_pixels = CorrelationEditor.scale_bar_size / world_pixel
            m = CorrelationEditor.SCALE_BAR_WINDOW_MARGIN
            window_width = 2 * m + scale_bar_pixels
            window_height = 2 * m + CorrelationEditor.SCALE_BAR_HEIGHT
            imgui.set_next_window_size(window_width, window_height)
            world_pos = self.camera.world_to_screen_position(CorrelationEditor.scale_bar_world_position)
            imgui.set_next_window_position(world_pos[0], world_pos[1])
            imgui.begin("##scalebar", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_BACKGROUND | imgui.WINDOW_NO_MOVE)
            draw_list = imgui.get_background_draw_list()
            window_pos = imgui.get_window_position()
            draw_list.add_rect_filled(window_pos[0] + m, window_pos[1] + m, window_pos[0] + scale_bar_pixels + m, window_pos[1] + CorrelationEditor.SCALE_BAR_HEIGHT + m, imgui.get_color_u32_rgba(*CorrelationEditor.SCALE_BAR_COLOUR))
            if imgui.begin_popup_context_window():
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1.0)
                _c, CorrelationEditor.SCALE_BAR_COLOUR = imgui.color_edit4("Scale bar colour", *CorrelationEditor.SCALE_BAR_COLOUR, flags=imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                imgui.pop_style_var(1)
                imgui.end_popup()
            elif imgui.is_mouse_down(glfw.MOUSE_BUTTON_LEFT) and imgui.is_window_focused():
                cursor_world_pos_prev = self.camera.cursor_to_world_position(self.window.cursor_pos_previous_frame)
                cursor_world_pos_current = self.camera.cursor_to_world_position(self.window.cursor_pos)
                delta_x = cursor_world_pos_current[0] - cursor_world_pos_prev[0]
                delta_y = cursor_world_pos_current[1] - cursor_world_pos_prev[1]
                CorrelationEditor.scale_bar_world_position[0] += delta_x
                CorrelationEditor.scale_bar_world_position[1] += delta_y
            imgui.end()
            imgui.pop_style_color(2)
            imgui.pop_style_var(3)

    def visuals_window(self):
        imgui.begin("##visualsw", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_BACKGROUND | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, cfg.CE_WIDGET_ROUNDING)
        imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, cfg.CE_WIDGET_ROUNDING)
        imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1.0)

        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_PANEL_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *cfg.COLOUR_PANEL_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_PANEL_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *cfg.COLOUR_TITLE_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *cfg.COLOUR_TITLE_BACKGROUND)

        if CorrelationEditor.active_frame is not None:
            if CorrelationEditor.active_frame.has_slices:
                imgui.set_next_item_width(CorrelationEditor.ALPHA_SLIDER_WIDTH + CorrelationEditor.BLEND_COMBO_WIDTH + 35.0)
                _c, requested_slice = imgui.slider_int("##slicer", CorrelationEditor.active_frame.current_slice, 0, CorrelationEditor.active_frame.n_slices, format=f"slice {CorrelationEditor.active_frame.current_slice+1:.0f}/{CorrelationEditor.active_frame.n_slices:.0f}")
                if _c:
                    CorrelationEditor.active_frame.set_slice(requested_slice)
            else:
                imgui.dummy(0.0, 19.0)
            # interpolation mode
            interp = CorrelationEditor.active_frame.interpolate
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0.0, 0.0))
            imgui.push_style_color(imgui.COLOR_BUTTON, *(1.0, 1.0, 1.0, 0.7))
            if imgui.image_button(self.icon_linterp.renderer_id if interp else self.icon_no_interp.renderer_id, 19, 19):
                CorrelationEditor.active_frame.toggle_interpolation()
            CorrelationEditor.tooltip("Toggle interpolation mode")
            imgui.pop_style_color(1)
            imgui.pop_style_var(1)
            imgui.same_line()

            imgui.set_next_item_width(CorrelationEditor.ALPHA_SLIDER_WIDTH)
            _, CorrelationEditor.active_frame.alpha = imgui.slider_float("##alpha", CorrelationEditor.active_frame.alpha, 0.0, 1.0, format="alpha = %.2f")

            # blend mode
            imgui.same_line()
            imgui.set_next_item_width(CorrelationEditor.BLEND_COMBO_WIDTH)
            _, CorrelationEditor.active_frame.blend_mode = imgui.combo("##blending", CorrelationEditor.active_frame.blend_mode, CorrelationEditor.BLEND_MODES_LIST)
        imgui.pop_style_color(5)
        imgui.pop_style_var(3)
        imgui.end()

    def camera_control(self):
        if imgui.get_io().want_capture_mouse or imgui.get_io().want_capture_keyboard:
            return None
        if self.window.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE):
            delta_cursor = self.window.cursor_delta
            self.camera.position[0] += delta_cursor[0] / self.camera.zoom
            self.camera.position[1] -= delta_cursor[1] / self.camera.zoom
        if self.window.get_key(glfw.KEY_LEFT_SHIFT):
            self.camera.zoom *= (1.0 + self.window.scroll_delta[1] * CorrelationEditor.CAMERA_ZOOM_STEP)
            self.camera.zoom = min([self.camera.zoom, CorrelationEditor.CAMERA_MAX_ZOOM])

    def snap_frame(self, cursor_world_pos):
        if not CorrelationEditor.snap_enabled:
            return False, 0, 0
        # snap only if cursor is close to one of the corners of the active frame.
        corners = CorrelationEditor.active_frame.corner_positions
        p = copy(cursor_world_pos)
        d = list()

        for xy in corners:
            d.append(np.sqrt((xy[0] - p[0])**2 + (xy[1] - p[1])**2))
        idx = np.argmin(d)

        source = corners[idx]  # source is now the corner position of the corner closest to the cursor
        if d[idx] > (CorrelationEditor.SNAPPING_ACTIVATION_DISTANCE * CorrelationEditor.DEFAULT_ZOOM / self.camera.zoom):
            return False, 0, 0

        children = CorrelationEditor.active_frame.list_all_children(include_self=True)
        for frame in cfg.ce_frames:
            if frame in children:
                continue
            if frame.hide:
                continue
            corners = frame.corner_positions
            for target in corners:
                distance = np.sqrt((target[0] - source[0])**2 + (target[1] - source[1])**2)
                if distance < (CorrelationEditor.SNAPPING_DISTANCE * CorrelationEditor.DEFAULT_ZOOM / self.camera.zoom):
                    return True, target[0] - source[0], target[1] - source[1]
        return False, 0, 0

    def editor_control(self):
        if imgui.get_io().want_capture_mouse:
            self.window.mouse_event = None
        if imgui.get_io().want_capture_keyboard:
            self.window.mouse_event = None
            self.window.key_event = None
            return
        if not self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT):
            CorrelationEditor.active_gizmo = None
        if CorrelationEditor.frames_dropped:
            return
        if CorrelationEditor.measure_active:
            CorrelationEditor.measure_tool.on_update()
            # measure tool input
            world_point = self.camera.cursor_to_world_position(self.window.cursor_pos)
            if self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.PRESS):
                CorrelationEditor.measure_tool.set_point(world_point)
            else:
                CorrelationEditor.measure_tool.hover_pos = world_point
            return
        # Editor mouse input - selecting frames and changing frame gizmo mode
        if self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.PRESS):
            # check which object should be active (if any)
            clicked_object = self.get_object_under_cursor(self.window.cursor_pos)
            if isinstance(clicked_object, CLEMFrame):
                if CorrelationEditor.active_frame != clicked_object:
                    CorrelationEditor.gizmo_mode_scale = True
                    CorrelationEditor.active_frame_timer = 0.0
                CorrelationEditor.active_frame = clicked_object
                CorrelationEditor.active_gizmo = None
            elif isinstance(clicked_object, EditorGizmo):
                CorrelationEditor.active_gizmo = clicked_object
            else:
                CorrelationEditor.active_gizmo = None
                CorrelationEditor.active_frame = None
            self.mouse_left_press_world_pos = self.camera.cursor_to_world_position(self.window.cursor_pos)
        elif self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, max_duration=min([CorrelationEditor.active_frame_timer, CorrelationEditor.MOUSE_SHORT_PRESS_MAX_DURATION])):
            clicked_object = self.get_object_under_cursor(self.window.cursor_pos, prioritize_active_frame=False)
            if CorrelationEditor.active_frame == clicked_object:
                CorrelationEditor.gizmo_mode_scale = not CorrelationEditor.gizmo_mode_scale
            elif isinstance(clicked_object, CLEMFrame):
                CorrelationEditor.active_frame = clicked_object

        # Editor mouse input - dragging frames and gizmo's
        if not imgui.get_io().want_capture_mouse:  # skip mouse drag input when imgui expects mouse input
            cursor_world_pos_now = self.camera.cursor_to_world_position(self.window.cursor_pos)
            cursor_world_pos_prev = self.camera.cursor_to_world_position(self.window.cursor_pos_previous_frame)
            if CorrelationEditor.active_gizmo is not None and CorrelationEditor.active_gizmo.gizmo_type == EditorGizmo.TYPE_LOCATION:
                delta_x = cursor_world_pos_now[0] - cursor_world_pos_prev[0]
                delta_y = cursor_world_pos_now[1] - cursor_world_pos_prev[1]
                CorrelationEditor.active_gizmo.transform.translation[0] += delta_x
                CorrelationEditor.active_gizmo.transform.translation[1] += delta_y
            elif CorrelationEditor.active_gizmo is None and CorrelationEditor.active_frame is not None:
                if self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT):  # user is dragging the active frame
                    snap, delta_x, delta_y = self.snap_frame(cursor_world_pos_now)
                    if not snap:
                        delta_x = cursor_world_pos_now[0] - cursor_world_pos_prev[0]
                        delta_y = cursor_world_pos_now[1] - cursor_world_pos_prev[1]
                    for frame in CorrelationEditor.active_frame.list_all_children(include_self=True):
                        frame.translate([delta_x, delta_y])
            elif CorrelationEditor.active_frame is not None:
                pivot = CorrelationEditor.active_frame.pivot_point
                if self.window.get_key(glfw.KEY_LEFT_SHIFT):  # if LEFT_SHIFT pressed, rotate/scale is around center of image instead of around pivot.
                    pivot = CorrelationEditor.active_frame.transform.translation
                if CorrelationEditor.active_gizmo.gizmo_type == EditorGizmo.TYPE_PIVOT:
                    # Find the world position delta between prev and current frame and apply to pivot
                    delta_x = cursor_world_pos_now[0] - cursor_world_pos_prev[0]
                    delta_y = cursor_world_pos_now[1] - cursor_world_pos_prev[1]
                    CorrelationEditor.active_frame.pivot_point[0] += delta_x
                    CorrelationEditor.active_frame.pivot_point[1] += delta_y
                elif CorrelationEditor.active_gizmo.gizmo_type == EditorGizmo.TYPE_ROTATE:
                    # Find the angle moved by cursor between prev and current frame and apply a pivoted rotation
                    angle_i = np.arctan2(cursor_world_pos_prev[0] - pivot[0], cursor_world_pos_prev[1] - pivot[1])
                    angle_f = np.arctan2(cursor_world_pos_now[0] - pivot[0], cursor_world_pos_now[1] - pivot[1])
                    delta_angle = (angle_f - angle_i) / np.pi * 180.0
                    for frame in CorrelationEditor.active_frame.list_all_children(include_self=True):
                        frame.pivoted_rotation(pivot, -delta_angle)
                elif CorrelationEditor.active_gizmo.gizmo_type == EditorGizmo.TYPE_SCALE:
                    # Find the scaling
                    offset_vec_i = [pivot[0] - cursor_world_pos_prev[0], pivot[1] - cursor_world_pos_prev[1]]
                    offset_vec_f = [pivot[0] - cursor_world_pos_now[0], pivot[1] - cursor_world_pos_now[1]]
                    scale_i = np.sqrt(offset_vec_i[0] ** 2 + offset_vec_i[1] ** 2)
                    scale_f = np.sqrt(offset_vec_f[0] ** 2 + offset_vec_f[1] ** 2)
                    relative_scale = 1.0
                    if scale_i != 0.0:
                        relative_scale = scale_f / scale_i
                    for frame in CorrelationEditor.active_frame.list_all_children(include_self=True):
                        frame.pivoted_scale(pivot, relative_scale)

        # Editor mouse input - right-click options menu
        if self.window.get_mouse_event(glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS):
            self.context_menu_obj = self.get_object_under_cursor(self.window.cursor_pos)
            self.context_menu_position = copy(self.window.cursor_pos)
            self.context_menu_open = True

        # Editor key input
        if CorrelationEditor.active_frame is not None:
            ctrl = imgui.is_key_down(glfw.KEY_LEFT_CONTROL) or imgui.is_key_down(glfw.KEY_RIGHT_CONTROL)
            shift = imgui.is_key_down(glfw.KEY_LEFT_SHIFT) or imgui.is_key_down(glfw.KEY_RIGHT_SHIFT)
            translation_step = CorrelationEditor.ARROW_KEY_TRANSLATION
            if ctrl: translation_step = CorrelationEditor.ARROW_KEY_TRANSLATION_SLOW
            if shift: translation_step = CorrelationEditor.ARROW_KEY_TRANSLATION_FAST
            if imgui.is_key_pressed(glfw.KEY_LEFT, repeat=True):
                CorrelationEditor.active_frame.transform.translation[0] -= translation_step
            elif imgui.is_key_pressed(glfw.KEY_RIGHT, repeat=True):
                CorrelationEditor.active_frame.transform.translation[0] += translation_step
            elif imgui.is_key_pressed(glfw.KEY_UP, repeat=True):
                CorrelationEditor.active_frame.transform.translation[1] += translation_step
            elif imgui.is_key_pressed(glfw.KEY_DOWN, repeat=True):
                CorrelationEditor.active_frame.transform.translation[1] -= translation_step
            elif imgui.is_key_pressed(glfw.KEY_DELETE, repeat=True):
                if CorrelationEditor.active_frame is not None:
                    CorrelationEditor.delete_frame(CorrelationEditor.active_frame)

    def _warning_window(self):
        def ww_context_menu():
            imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *cfg.COLOUR_MENU_WINDOW_BACKGROUND)
            if imgui.begin_popup_context_window():
                raise_error, _ = imgui.menu_item("Raise error (debug)")
                if raise_error:
                    raise cfg.error_obj
                copy_error, _ = imgui.menu_item("Copy to clipboard")
                if copy_error:
                    pyperclip.copy(cfg.error_msg)
                imgui.end_popup()
            imgui.pop_style_color(1)
        ## Error message
        if cfg.error_msg is not None:
            if cfg.error_new:
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *cfg.COLOUR_ERROR_WINDOW_HEADER_NEW)
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *cfg.COLOUR_ERROR_WINDOW_HEADER_NEW)
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *cfg.COLOUR_ERROR_WINDOW_HEADER_NEW)
            else:
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *cfg.COLOUR_ERROR_WINDOW_HEADER)
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *cfg.COLOUR_ERROR_WINDOW_HEADER)
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *cfg.COLOUR_ERROR_WINDOW_HEADER)
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *cfg.COLOUR_ERROR_WINDOW_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_ERROR_WINDOW_TEXT)
            imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 3.0)
            imgui.set_next_window_size(self.window.width, cfg.ERROR_WINDOW_HEIGHT)
            imgui.set_next_window_position(0, self.window.height - cfg.ERROR_WINDOW_HEIGHT)
            _, stay_open = imgui.begin("Warning", True, imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE)
            imgui.text(cfg.error_msg)
            if imgui.is_window_focused() and self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.PRESS):
                cfg.error_new = False
            ww_context_menu()
            imgui.end()
            if not stay_open:
                cfg.error_msg = None
                cfg.error_new = True
            imgui.pop_style_color(5)
            imgui.pop_style_var(1)

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
                elif imgui.menu_item("Rotate +90° (ccw)")[0]:
                    frame.transform.rotation += 90.0
                elif imgui.menu_item("Rotate -90° (cw)")[0]:
                    frame.transform.rotation -= 90.0
                elif imgui.menu_item("Flip horizontally")[0]:
                    frame.flip()
                elif imgui.menu_item("Flip vertically")[1]:
                    frame.flip(horizontally=False)
                if imgui.begin_menu("Render binned"):
                    if imgui.menu_item("None", selected=frame.binning==1)[0]:
                        frame.binning = 1
                    if imgui.menu_item("1.5 x", selected=frame.binning==1.5)[0]:
                        frame.binning = 1.5
                    elif imgui.menu_item("2 x", selected=frame.binning==2)[0]:
                        frame.binning = 2
                    elif imgui.menu_item("3 x", selected=frame.binning==3)[0]:
                        frame.binning = 3
                    elif imgui.menu_item("4 x", selected=frame.binning==4)[0]:
                        frame.binning = 4
                    elif imgui.menu_item("8 x", selected=frame.binning==8)[0]:
                        frame.binning = 8
                    imgui.end_menu()
                if self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT) and not imgui.is_window_hovered(imgui.HOVERED_CHILD_WINDOWS | imgui.HOVERED_ALLOW_WHEN_BLOCKED_BY_POPUP):
                    self.context_menu_open = False
                imgui.end()

    def end_frame(self):
        self.window.end_frame()

    def get_object_under_cursor(self, cursor_position, prioritize_active_frame=True):
        """In this function, the cursor position is first translated into the corresponding world position and
        subsequently it is checked whether that world position lies within the quad of any of the existing CLEMFrame
        or EditorGizmo objects. Gizmos have priority."""
        def is_point_in_rectangle(point, corner_positions):
            P = point
            A = corner_positions[0]
            B = corner_positions[1]
            D = corner_positions[3]
            ap = [P[0] - A[0], P[1] - A[1]]
            ab = [B[0] - A[0], B[1] - A[1]]
            ad = [D[0] - A[0], D[1] - A[1]]
            return (0 < ap[0] * ab[0] + ap[1] * ab[1] < ab[0]**2 + ab[1]**2) and (0 < ap[0] * ad[0] + ap[1] * ad[1] < ad[0]**2 + ad[1]**2)

        cursor_world_position = self.camera.cursor_to_world_position(cursor_position)
        for gizmo in CorrelationEditor.gizmos:
            if gizmo.hide:
                continue
            if is_point_in_rectangle(cursor_world_position, gizmo.corner_positions):
                return gizmo
        if prioritize_active_frame:
            if CorrelationEditor.active_frame is not None:
                if is_point_in_rectangle(cursor_world_position, CorrelationEditor.active_frame.corner_positions):
                    return CorrelationEditor.active_frame
        for frame in cfg.ce_frames:
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

    @staticmethod
    def delete_frame(frame):
        for child in frame.children:
            child.parent_to(frame.parent)
        frame.unparent()
        if CorrelationEditor.active_frame == frame:
            CorrelationEditor.active_frame = None
        if CorrelationEditor.frame_drag_payload == frame:
            CorrelationEditor.frame_drag_payload = None
        if frame in cfg.ce_frames:
            cfg.ce_frames.remove(frame)

    @staticmethod
    def add_frame(frame_obj):
        CorrelationEditor.incoming_frame_buffer.append(frame_obj)

    @staticmethod
    def init_toolkit():
        class ToolImpl:
            def __init__(self, tool_create_fn):
                self.create_fn = tool_create_fn
                self.tool_obj = tool_create_fn()
                self.title = self.tool_obj.title
                self.info = self.tool_obj.description

        toolimpls = list()
        tool_source_files = glob.glob("ceplugins/*.py")
        for toolsrc in tool_source_files:
            if "custom_tool_template" in toolsrc or "__init__.py" in toolsrc:
                continue

            module_name = toolsrc[toolsrc.rfind("\\")+1:-3]
            try:
                mod = importlib.import_module("ceplugins."+module_name)
                toolimpls.append(ToolImpl(mod.create))
            except Exception as e:
                cfg.set_error(e, f"No well-defined Tool type found in {toolsrc}. See manual for minimal code requirements.")

        CorrelationEditor.tool_factory = dict()
        CorrelationEditor.tool_descriptions = list()
        for toolimpl in toolimpls:
            CorrelationEditor.tool_factory[toolimpl.title] = toolimpl.create_fn
            CorrelationEditor.tool_descriptions.append(toolimpl.info)
        CorrelationEditor.tools_list = list(CorrelationEditor.tool_factory.keys())


class Renderer:
    def __init__(self):
        self.quad_shader = Shader("shaders/ce_quad_shader.glsl")
        self.border_shader = Shader("shaders/ce_border_shader.glsl")
        self.gizmo_shader = Shader("shaders/ce_gizmo_shader.glsl")

    def render_frame_quad(self, camera, frame, override_blending=False):
        if override_blending:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBlendEquation(GL_FUNC_ADD)
        else:
            blend_mode = CorrelationEditor.BLEND_MODES[CorrelationEditor.BLEND_MODES_LIST[frame.blend_mode]]
            glBlendFunc(blend_mode[0], blend_mode[1])
            glBlendEquation(blend_mode[2])
        if not frame.hide:
            self.quad_shader.bind()
            frame.quad_va.bind()
            frame.texture.bind(0)
            frame.lut_texture.bind(1)
            self.quad_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
            self.quad_shader.uniform1f("alpha", frame.alpha)
            self.quad_shader.uniformmat4("modelMatrix", frame.transform.matrix)
            if frame.is_rgb:
                self.quad_shader.uniform1f("rgbMode", 1.0)
                l = frame.rgb_contrast_lims
                self.quad_shader.uniform3f("contrastMin", [l[0] / 255.0, l[2] / 255.0, l[4] / 255.0])
                self.quad_shader.uniform3f("contrastMax", [l[1] / 255.0, l[3] / 255.0, l[5] / 255.0])
            else:
                self.quad_shader.uniform1f("rgbMode", 0.0)
                self.quad_shader.uniform3f("contrastMin", [frame.contrast_lims[0], 0.0, 0.0])
                self.quad_shader.uniform3f("contrastMax", [frame.contrast_lims[1], 0.0, 0.0])
            self.quad_shader.uniform1f("hpix", frame.width)
            self.quad_shader.uniform1f("vpix", frame.height)
            self.quad_shader.uniform1f("binning", frame.binning)
            glDrawElements(GL_TRIANGLES, frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            self.quad_shader.unbind()
            frame.quad_va.unbind()
            glActiveTexture(GL_TEXTURE0)

    def render_frame_border(self, camera, frame):
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR)
        glBlendEquation(GL_FUNC_ADD)
        self.border_shader.bind()
        frame.border_va.bind()
        self.border_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.border_shader.uniformmat4("modelMatrix", frame.transform.matrix)
        self.border_shader.uniform3f("lineColour", CorrelationEditor.COLOUR_IMAGE_BORDER)
        glDrawElements(GL_LINES, frame.border_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.border_shader.unbind()
        frame.border_va.unbind()

    def render_measure_tool(self, camera, tool):
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)
        if not tool.p_set:
            return
        self.border_shader.bind()
        tool.va.bind()
        self.border_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.border_shader.uniformmat4("modelMatrix", np.identity(4))
        self.border_shader.uniform3f("lineColour", CorrelationEditor.measure_tool_colour)
        glDrawElements(GL_LINES, tool.va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.border_shader.unbind()
        tool.va.unbind()

    def render_roi(self, camera, roi):
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)
        self.border_shader.bind()
        roi.va.bind()
        self.border_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.border_shader.uniformmat4("modelMatrix", np.identity(4))
        self.border_shader.uniform3f("lineColour", CorrelationEditor.EXPORT_ROI_LINE_COLOUR)
        glDrawElements(GL_LINES, roi.va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.border_shader.unbind()
        roi.va.unbind()

    def render_gizmo(self, camera, gizmo):
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR)
        glBlendEquation(GL_FUNC_ADD)
        if not gizmo.hide:
            self.gizmo_shader.bind()
            gizmo.va.bind()
            gizmo.texture.bind(0)
            self.gizmo_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
            self.gizmo_shader.uniformmat4("modelMatrix", gizmo.transform.matrix)
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
        self.projection_width = 1
        self.projection_height = 1
        self.set_projection_matrix(cfg.window_width, cfg.window_height)

    def cursor_to_world_position(self, cursor_pos):
        """Converts an input cursor position to corresponding world position. Assuming orthographic projection matrix."""
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
            [0.0, 0.0, self.zoom, self.position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.view_projection_matrix = np.matmul(self.projection_matrix, self.view_matrix)


class EditorGizmo:
    idgen = count(0)
    ICON_SIZE = 10.0
    TYPE_SCALE = 0
    TYPE_ROTATE = 1
    TYPE_PIVOT = 2
    TYPE_LOCATION = 3

    ICON_TEXTURES = dict()
    ICON_SIZES = dict()
    ICON_SIZES[TYPE_SCALE] = 1.0
    ICON_SIZES[TYPE_ROTATE] = 0.8
    ICON_SIZES[TYPE_PIVOT] = 0.66
    ICON_SIZES[TYPE_LOCATION] = 1.0

    ICON_OFFSETS = dict()
    ICON_OFFSETS[TYPE_SCALE] = 1000.0
    ICON_OFFSETS[TYPE_ROTATE] = 1500.0
    ICON_OFFSETS[TYPE_PIVOT] = 0.0
    ICON_OFFSETS[TYPE_LOCATION] = 0.0

    @staticmethod
    def init_textures():
        icon_scale = np.asarray(Image.open("icons/icon_scale_256.png")).astype(np.float) / 255.0
        icon_rotate = np.asarray(Image.open("icons/icon_rotate_256.png")).astype(np.float) / 255.0
        icon_pivot = np.asarray(Image.open("icons/icon_pivot_256.png")).astype(np.float) / 255.0
        icon_location = np.asarray(Image.open("icons/icon_location_256.png")).astype(np.float) / 255.0
        icon_scale[:, :, 0:2] = 1.0
        icon_rotate[:, :, 0:2] = 1.0
        icon_pivot[:, :, 0:2] = 1.0
        icon_location[:, :, 0:2] = 1.0
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_SCALE] = Texture(format="rgba32f")
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_SCALE].update(icon_scale)
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_SCALE].set_linear_interpolation()
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_ROTATE] = Texture(format="rgba32f")
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_ROTATE].update(icon_rotate)
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_ROTATE].set_linear_interpolation()
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_PIVOT] = Texture(format="rgba32f")
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_PIVOT].update(icon_pivot)
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_PIVOT].set_linear_interpolation()
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_LOCATION] = Texture(format="rgba32f")
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_LOCATION].update(icon_location)
        EditorGizmo.ICON_TEXTURES[EditorGizmo.TYPE_LOCATION].set_linear_interpolation()

    def __init__(self, gizmo_type=TYPE_SCALE, idx=None):
        self.uid = next(EditorGizmo.idgen)
        self.gizmo_type = gizmo_type
        self.texture = EditorGizmo.ICON_TEXTURES[self.gizmo_type]
        self.va = VertexArray()
        self.corner_positions_local = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.corner_positions = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.idx = idx
        self.generate_va()
        self.hide = False
        self.transform = Transform()
        self.camera_zoom = 0.0

    def generate_va(self):
        w = EditorGizmo.ICON_SIZE * EditorGizmo.ICON_SIZES[self.gizmo_type]
        vertex_attributes = [-w, w, 1.0, 0.0, 1.0,
                             -w, -w, 1.0, 0.0, 0.0,
                             w, -w, 1.0, 1.0, 0.0,
                             w, w, 1.0, 1.0, 1.0]
        self.corner_positions_local = [[-w, w], [-w, -w], [w, -w], [w, w]]
        indices = [0, 1, 2, 2, 0, 3]
        self.va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))

    def set_parent_frame(self, frame, camera_zoom):
        # set the gizmo's positions in accordance with the parent frame. Also input camera zoom to adjust gizmo icon offset.
        if self.gizmo_type == EditorGizmo.TYPE_PIVOT:
            self.transform.translation = copy(frame.pivot_point)
        elif self.gizmo_type == EditorGizmo.TYPE_LOCATION:
            pass
        else:
            frame_corners = deepcopy(frame.corner_positions)
            self.transform.translation = (frame_corners[self.idx])
            self.transform.rotation = frame.transform.rotation + self.idx * 90.0
            offset = CorrelationEditor.DEFAULT_WORLD_PIXEL_SIZE * EditorGizmo.ICON_SIZE * CorrelationEditor.DEFAULT_ZOOM / camera_zoom
            self.transform.translation[0] += offset * -np.sin((self.idx * 2 + 1) * np.pi / 4 + frame.transform.rotation / 180.0 * np.pi)
            self.transform.translation[1] += offset * np.cos((self.idx * 2 + 1) * np.pi / 4 + frame.transform.rotation / 180.0 * np.pi)
        self.transform.scale = 0.0 if self.camera_zoom == 0.0 else 1.0 / self.camera_zoom
        self.transform.compute_matrix()

        for i in range(4):
            local_corner_pos = tuple(self.corner_positions_local[i])
            vec = np.matrix([*local_corner_pos, 0.0, 1.0]).T
            world_corner_pos = self.transform.matrix * vec
            self.corner_positions[i] = [float(world_corner_pos[0]), float(world_corner_pos[1])]

    def set_no_parent_frame(self, camera_zoom):
        self.transform.scale = 0.0 if self.camera_zoom == 0.0 else 1.0 / self.camera_zoom
        self.transform.compute_matrix()

        for i in range(4):
            local_corner_pos = tuple(self.corner_positions_local[i])
            vec = np.matrix([*local_corner_pos, 0.0, 1.0]).T
            world_corner_pos = self.transform.matrix * vec
            self.corner_positions[i] = [float(world_corner_pos[0]), float(world_corner_pos[1])]

    def set_zoom_compensation_factor(self, camera_zoom):
        self.camera_zoom = camera_zoom


class ImportedFrameData:
    def __init__(self, path):
        self.path = path
        self.title = path[path.rfind("\\")+1:]
        self.extension = path[path.rfind("."):]
        self.pxd = None
        self.n_slices = 1
        self.pixel_size = CorrelationEditor.DEFAULT_IMAGE_PIXEL_SIZE
        try:
            if self.extension == ".tiff" or self.extension == ".tif":
                img = Image.open(path)
                self.pxd = np.asarray(img)
                if CorrelationEditor.flip_images_on_load:
                    self.pxd = np.flip(self.pxd, axis=0)
                self.n_slices = img.n_frames
            elif self.extension == ".png":
                self.pxd = np.asarray(Image.open(path)) / 255.0
                if CorrelationEditor.flip_images_on_load:
                    self.pxd = np.flip(self.pxd, axis=0)
            elif self.extension == ".mrc":
                with mrcfile.open(self.path) as mrc:
                    self.pixel_size = float(mrc.voxel_size.x / 10.0)
                mrc = mrcfile.mmap(self.path, mode="r")
                if len(mrc.data.shape) == 2:
                    self.pxd = mrc.data
                else:
                    if CorrelationEditor.mrc_flip_on_load:
                        self.pxd = mrc.data[:, 0, :]
                        self.n_slices = mrc.data.shape[1]
                    else:
                        self.pxd = mrc.data[0, :, :]
                        self.n_slices = mrc.data.shape[0]
        except Exception as e:
            pass

    def to_CLEMFrame(self):
        if self.pxd is None:
            return False
        clem_frame = CLEMFrame(self.pxd)
        clem_frame.title = self.title
        clem_frame.path = self.path
        clem_frame.extension = self.extension
        clem_frame.pixel_size = self.pixel_size
        if self.n_slices > 1:
            clem_frame.has_slices = True
            clem_frame.n_slices = self.n_slices
        return clem_frame


class MeasureTool:
    def __init__(self):
        """Measure tool is just a line from point p to point q with distance |pq| indicated."""
        self.p = np.array([0.0, 0.0])
        self.q = np.array([0.0, 0.0])
        self.currently_editing_pos_p = True
        self.q_set = False
        self.p_set = False
        self.distance = 1
        self.hover_pos = np.array([0.0, 0.0])
        self.va = VertexArray(attribute_format="xy")

    def on_update(self):
        if self.q_set:
            vertex_attributes = [*self.p, *self.q]
        else:
            vertex_attributes = [*self.p, *self.hover_pos]
        self.distance = np.sqrt((vertex_attributes[0] - vertex_attributes[2]) ** 2 + (vertex_attributes[1] - vertex_attributes[3]) ** 2)
        indices = [0, 1]
        self.va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))

    def set_point(self, point):
        if self.currently_editing_pos_p:
            self.currently_editing_pos_p = False
            self.p = np.array(point)
            self.q_set = False
            self.p_set = True
        else:
            self.currently_editing_pos_p = True
            self.q = np.array(point)
            self.q_set = True

    def reset(self):
        self.p_set = False
        self.q_set = False


class ExportROI:
    def __init__(self):
        self.roi = [-1, -1, 1, 1]
        self.va = VertexArray(attribute_format="xy")
        self.set_roi(self.roi)

    def set_roi(self, roi):
        self.roi = roi
        r = self.roi
        vertex_attributes = [r[0], r[1], r[2], r[1], r[2], r[3], r[0], r[3]]
        indices = [0, 1, 1, 2, 2, 3, 3, 0]
        self.va.update(VertexBuffer(vertex_attributes), IndexBuffer(indices))
