import glfw
import imgui
from PIL import Image
from copy import copy
from tkinter import filedialog
import dill as pickle
from scNodes.core.se_model import *
from scNodes.core.se_frame import *
import scNodes.core.widgets as widgets
from scNodes.core.util import clamp, bin_mrc, get_maxima_3d_watershed
import pyperclip
import os

## TODO: U2OS tomo dataset: int8 of uint8? clipping als naar float


class SegmentationEditor:
    if True:
        CAMERA_ZOOM_STEP = 0.1
        CAMERA_MAX_ZOOM = 100.0
        DEFAULT_HORIZONTAL_FOV_WIDTH = 1000
        DEFAULT_ZOOM = 1.0  # adjusted in init
        DEFAULT_WORLD_PIXEL_SIZE = 1.0  # adjusted on init

        # GUI params
        MAIN_WINDOW_WIDTH = 330
        FEATURE_PANEL_HEIGHT = 104
        INFO_HISTOGRAM_HEIGHT = 70
        SLICER_WINDOW_VERTICAL_OFFSET = 30
        SLICER_WINDOW_WIDTH = 700
        ACTIVE_SLICES_CHILD_HEIGHT = 140
        PROGRESS_BAR_HEIGHT = 8
        MODEL_PANEL_HEIGHT_TRAINING = 158
        MODEL_PANEL_HEIGHT_PREDICTION = 145
        MODEL_PANEL_HEIGHT_LOGIC = 115

        TOOLTIP_APPEAR_DELAY = 1.0
        TOOLTIP_HOVERED_TIMER = 0.0
        TOOLTIP_HOVERED_START_TIME = 0.0

        renderer = None

        BLEND_MODES = dict()  # blend mode template: (glBlendFunc ARG1, ... ARG2, glBlendEquation ARG1, glsl_side_blend_mode_code)
        BLEND_MODES["Sum"] = (GL_SRC_ALPHA, GL_DST_ALPHA, GL_FUNC_ADD, 0)
        BLEND_MODES["Colourize"] = (GL_DST_COLOR, GL_DST_ALPHA, GL_FUNC_ADD, 1)
        BLEND_MODES["Mask"] = (GL_ZERO, GL_SRC_ALPHA, GL_FUNC_ADD, 2)
        BLEND_MODES["Overlay"] = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_FUNC_ADD, 3)
        BLEND_MODES_3D = dict()  # blend mode template: (glBlendFunc ARG1, ... ARG2, glBlendEquation ARG1, glsl_side_blend_mode_code)
        BLEND_MODES_3D["Halo"] = (GL_SRC_ALPHA, GL_DST_ALPHA, GL_FUNC_ADD, 0)
        BLEND_MODES_3D["Binary"] = (GL_SRC_ALPHA, GL_DST_ALPHA, GL_FUNC_ADD, 1)
        BLEND_MODES_LIST = list(BLEND_MODES.keys())
        BLEND_MODES_LIST_3D = list(BLEND_MODES_3D.keys())

        pick_tab_index_datasets_segs = False
        VIEW_3D_PIVOT_SPEED = 0.3
        VIEW_3D_MOVE_SPEED = 100.0
        PICKING_FRAME_ALPHA = 1.0
        SELECTED_RENDER_STYLE = 1
        RENDER_STYLES = ["Cartoon", "Phong", "Flat", "Misc."]
        RENDER_BOX = False
        RENDER_CLEAR_COLOUR = cfg.COLOUR_WINDOW_BACKGROUND[:3]
        RENDER_LIGHT_COLOUR = (1.0, 1.0, 1.0)
        VIEW_REQUIRES_UPDATE = True
        
        OVERLAY_ALPHA = 1.0
        OVERLAY_INTENSITY = 1.0
        OVERLAY_BLEND_MODE = 0
        OVERLAY_BLEND_MODE_3D = 0

        LIGHT_SPOT = None
        LIGHT_AMBIENT_STRENGTH = 0.5

        DATASETS_WINDOW_EXPANDED = False
        DATASETS_WINDOW_EXPANDED_HEIGHT = 500
        DATASETS_EXPORT_PANEL_EXPANDED = False
        DATASETS_EXPORT_PANEL_EXPANDED_HEIGHT = 300
        DATASETS_EXPORT_SELECT_ALL = False

        DATASETS_PICK_PANEL_EXPANDED = False
        DATASETS_PICK_PANEL_EXPANDED_HEIGHT = 300
        DATASETS_PICK_SELECT_ALL = False

        EXTRACT_THRESHOLD = 128
        EXTRACT_MIN_WEIGHT = 1000
        EXTRACT_MIN_SPACING = 10.0  # Angstrom
        EXTRACT_SELECTED_FEATURE_TITLE = ""

    def __init__(self, window, imgui_context, imgui_impl):
        self.window = window
        self.window.clear_color = cfg.COLOUR_WINDOW_BACKGROUND
        self.window.make_current()
        self.imgui_context = imgui_context
        self.imgui_implementation = imgui_impl

        self.camera = Camera()
        self.camera3d = Camera3D()
        SegmentationEditor.LIGHT_SPOT = Light3D()
        SegmentationEditor.DEFAULT_ZOOM = cfg.window_height / SegmentationEditor.DEFAULT_HORIZONTAL_FOV_WIDTH  # now it is DEFAULT_HORIZONTAL_FOV_WIDTH
        SegmentationEditor.DEFAULT_WORLD_PIXEL_SIZE = 1.0 / SegmentationEditor.DEFAULT_ZOOM
        self.camera.zoom = SegmentationEditor.DEFAULT_ZOOM
        SegmentationEditor.renderer = Renderer()
        self.filters = list()
        self.active_tab = "Segmentation"
        # training dataset params
        self.all_feature_names = list()
        self.feature_colour_dict = dict()
        self.trainset_feature_selection = dict()
        self.trainset_selection = list()
        self.trainset_num_boxes_positive = 0
        self.trainset_num_boxes_negative = 0
        self.trainset_boxsize = 64
        self.trainset_apix = 10.0
        self.show_trainset_boxes = False
        self.active_trainset_exports = list()

        # drop files
        self.incoming_files = list()

        # export & extract
        self.export_limit_range = True
        self.export_overlays = True
        self.export_dir = ""
        self.queued_exports = list()
        self.export_batch_size = 1
        self.queued_extracts = list()

        # crop handles
        self.crop_handles = list()

        # picking / 3d renders
        self.seg_folder = ""
        self.pick_same_folder = True
        self.pick_box_va = VertexArray(attribute_format="xyz")
        self.pick_box_quad_va = VertexArray(attribute_format="xyz")
        self.pick_box_va.update(VertexBuffer([0.0, 0.0]), IndexBuffer([]))
        self.pick_box_quad_va.update(VertexBuffer([0.0, 0.0]), IndexBuffer([]))
        for i in range(4):
            self.crop_handles.append(WorldSpaceIcon(i))

        if True:
            icon_dir = os.path.join(cfg.root, "icons")

            self.icon_close = Texture(format="rgba32f")
            pxd_icon_close = np.asarray(Image.open(os.path.join(icon_dir, "icon_close_256.png"))).astype(np.float32) / 255.0
            self.icon_close.update(pxd_icon_close)
            self.icon_close.set_linear_interpolation()

            self.icon_stop = Texture(format="rgba32f")
            pxd_icon_stop = np.asarray(Image.open(os.path.join(icon_dir, "icon_stop_256.png"))).astype(
                np.float32) / 255.0
            self.icon_stop.update(pxd_icon_stop)
            self.icon_stop.set_linear_interpolation()

            self.icon_obj = Texture(format="rgba32f")
            self.icon_blender = Texture(format="rgba32f")
            self.icon_chimerax = Texture(format="rgba32f")
            pxd = np.asarray(Image.open(os.path.join(icon_dir, "icon_obj_256.png"))).astype(np.float32) / 255.0
            print(pxd.shape)
            self.icon_obj.update(pxd)
            pxd = np.asarray(Image.open(os.path.join(icon_dir, "icon_blender_256.png"))).astype(np.float32) / 255.0
            print(pxd.shape)
            self.icon_blender.update(pxd)
            pxd = np.asarray(Image.open(os.path.join(icon_dir, "icon_chimerax_256.png"))).astype(np.float32) / 255.0
            print(pxd.shape)
            self.icon_chimerax.update(pxd)
            self.icon_obj.set_linear_interpolation()
            self.icon_chimerax.set_linear_interpolation()
            self.icon_blender.set_linear_interpolation()

    @staticmethod
    def set_active_dataset(dataset):
        SegmentationEditor.pick_tab_index_datasets_segs = True
        cfg.se_active_frame = dataset
        cfg.se_active_frame.requires_histogram_update = True
        cfg.se_active_frame.slice_changed = True
        SegmentationEditor.renderer.fbo1 = FrameBuffer(dataset.width, dataset.height, "rgba32f")
        SegmentationEditor.renderer.fbo2 = FrameBuffer(dataset.width, dataset.height, "rgba32f")
        SegmentationEditor.renderer.fbo3 = FrameBuffer(dataset.width, dataset.height, "rgba32f")
        if dataset.interpolate:
            SegmentationEditor.renderer.fbo1.texture.set_linear_interpolation()
            SegmentationEditor.renderer.fbo2.texture.set_linear_interpolation()
        else:
            SegmentationEditor.renderer.fbo1.texture.set_no_interpolation()
            SegmentationEditor.renderer.fbo2.texture.set_no_interpolation()

    def on_update(self):
        imgui.set_current_context(self.imgui_context)
        imgui.CONFIG_DOCKING_ENABLE = True  ## maayyyybe not.

        self.window.make_current()
        self.window.set_full_viewport()
        if self.window.focused:
            self.imgui_implementation.process_inputs()

        if not imgui.get_io().want_capture_keyboard and imgui.is_key_pressed(glfw.KEY_TAB):
            if imgui.is_key_down(glfw.KEY_LEFT_SHIFT):
                cfg.active_editor = (cfg.active_editor - 1) % len(cfg.editors)
            else:
                cfg.active_editor = (cfg.active_editor + 1) % len(cfg.editors)

        for filepath in self.window.dropped_files:
            self.import_dataset(filepath)

        self.window.on_update()

        if self.window.window_size_changed:
            cfg.window_width = self.window.width
            cfg.window_height = self.window.height
            self.camera.set_projection_matrix(cfg.window_width, cfg.window_height)
            self.camera3d.set_projection_matrix(cfg.window_width, cfg.window_height)

        if self.active_tab in ["Export", "Models"] and cfg.se_active_frame is not None and cfg.se_active_frame.slice_changed:
            emissions = list()
            absorbing_models = list()
            # launch the models
            for model in cfg.se_models:
                model.set_slice(cfg.se_active_frame.data, cfg.se_active_frame.pixel_size, cfg.se_active_frame.get_roi_indices(), cfg.se_active_frame.data.shape)
                if model.emit:
                    emissions.append(model.data)
                if model.absorb:
                    absorbing_models.append(model)
            # have models compete for pixels
            if len(emissions) > 0 and len(absorbing_models) > 0:
                # have models compete
                stacked_array = np.stack(emissions)
                maxima = np.max(stacked_array, axis=0)
                for model in absorbing_models:
                    model.data[model.data < maxima] = 0
            # apply interactions
            for model in cfg.se_models:
                for interaction in model.interactions:
                    interaction.apply(cfg.se_active_frame.pixel_size)
                model.update_texture()
            cfg.se_active_frame.slice_changed = False
        imgui.get_io().display_size = self.window.width, self.window.height
        imgui.new_frame()

        if self.queued_exports:
            if self.queued_exports[0].process.progress >= 1.0:
                self.queued_exports.pop(0)
                if self.queued_exports:
                    self.queued_exports[0].start()
        if self.queued_extracts:
            if self.queued_extracts[0].process.progress >= 1.0:
                self.queued_extracts.pop(0)
                if self.queued_extracts:
                    self.queued_extracts[0].start()
        # GUI calls
        if not self.window.is_minimized():
            self.camera_control()
            self.camera.on_update()
            self.camera3d.on_update()
            self.gui_main()
            SegmentationEditor.renderer.render_draw_list(self.camera)
            self.input()

        imgui.render()
        self.imgui_implementation.render(imgui.get_draw_data())
        imgui.CONFIG_DOCKING_ENABLE = False

    def input(self):
        if self.active_tab not in ["Segmentation", "Pick"] and cfg.se_active_frame is not None and cfg.se_active_frame.crop:
            SegmentationEditor.renderer.render_crop_handles(cfg.se_active_frame, self.camera, self.crop_handles)
        if cfg.se_active_frame is not None:
            sef = cfg.se_active_frame
            if imgui.is_key_pressed(glfw.KEY_C):
                sef.crop = not sef.crop
                if not sef.crop:
                    sef.crop_roi = [0, 0, sef.width, sef.height]
        if imgui.get_io().want_capture_mouse or imgui.get_io().want_capture_keyboard:
            return

        # key input
        active_frame = cfg.se_active_frame
        if active_frame in cfg.se_frames:
            if imgui.is_key_pressed(glfw.KEY_LEFT):
                idx = cfg.se_frames.index(active_frame) - 1
                idx = max(0, idx)
                SegmentationEditor.set_active_dataset(cfg.se_frames[idx])
            elif imgui.is_key_pressed(glfw.KEY_RIGHT):
                idx = cfg.se_frames.index(active_frame) + 1
                idx = min(idx, len(cfg.se_frames) - 1)
                SegmentationEditor.set_active_dataset(cfg.se_frames[idx])

        active_feature = None
        if active_frame is not None:
            active_feature = cfg.se_active_frame.active_feature

        # Key inputs that affect the active feature:
        if active_frame is not None:
            if not imgui.is_key_down(glfw.KEY_LEFT_SHIFT) and not imgui.is_key_down(
                    glfw.KEY_LEFT_CONTROL) and active_frame is not None:
                if self.window.scroll_delta[1] != 0.0:
                    idx = int(active_frame.current_slice - self.window.scroll_delta[1])
                    idx = idx % active_frame.n_slices
                    active_frame.set_slice(idx)

        if self.active_tab == "Segmentation":
            if active_feature is not None:
                if imgui.is_key_down(glfw.KEY_LEFT_CONTROL) and active_feature is not None:
                    active_feature.brush_size += self.window.scroll_delta[1]
                    active_feature.brush_size = max([1, active_feature.brush_size])
                if imgui.is_key_pressed(glfw.KEY_DOWN) or imgui.is_key_pressed(glfw.KEY_S):
                    idx = 0 if active_feature not in active_frame.features else active_frame.features.index(active_feature)
                    idx = (idx + 1) % len(active_frame.features)
                    cfg.se_active_frame.active_feature = cfg.se_active_frame.features[idx]
                elif imgui.is_key_pressed(glfw.KEY_UP) or imgui.is_key_pressed(glfw.KEY_W):
                    idx = 0 if active_feature not in active_frame.features else active_frame.features.index(active_feature)
                    idx = (idx - 1) % len(active_frame.features)
                    cfg.se_active_frame.active_feature = cfg.se_active_frame.features[idx]

            # Drawing / mouse input
            if active_feature is not None:
                cursor_world_position = self.camera.cursor_to_world_position(self.window.cursor_pos)
                pixel_coordinate = active_feature.parent.world_to_pixel_coordinate(cursor_world_position)


                if not imgui.is_key_down(glfw.KEY_LEFT_SHIFT):
                    if imgui.is_mouse_down(0):
                        Brush.apply_circular(active_feature, pixel_coordinate, True)
                        active_feature.hide = False
                    elif imgui.is_mouse_down(1):
                        Brush.apply_circular(active_feature, pixel_coordinate, False)
                else:
                    if imgui.is_mouse_clicked(0):
                        active_feature.add_box(pixel_coordinate)
                    elif imgui.is_mouse_clicked(1):
                        active_feature.remove_box(pixel_coordinate)
        elif self.active_tab != "Pick" and active_frame is not None and active_frame.crop:
            if not self.crop_handles[0].moving_entire_roi:
                any_handle_active = False
                for h in self.crop_handles:
                    if not h.active and h.is_hovered(self.camera, self.window.cursor_pos) and imgui.is_mouse_clicked(0):
                        h.active = True
                        any_handle_active = False
                    if h.active:
                        any_handle_active = True
                        world_pos = self.camera.cursor_to_world_position(self.window.cursor_pos)
                        pixel_pos = active_frame.world_to_pixel_coordinate(world_pos)
                        h.affect_crop(pixel_pos)
                        if not imgui.is_mouse_down(0):
                            h.active = False
                            h.convert_crop_roi_to_integers()
                            cfg.se_active_frame.slice_changed = True  # not really, but this flag is also used to trigger a model update.
                if not any_handle_active and imgui.is_mouse_clicked(0):
                    self.crop_handles[0].moving_entire_roi = True
                    self.crop_handles[0].convert_crop_roi_to_integers()
            else:
                if not imgui.is_mouse_down(0):
                    self.crop_handles[0].moving_entire_roi = False
                    self.crop_handles[0].convert_crop_roi_to_integers()
                    cfg.se_active_frame.slice_changed = True  # not really, but this flag is also used to trigger a model update.
                else:
                    world_pos_old = self.camera.cursor_to_world_position(self.window.cursor_pos_previous_frame)
                    world_pos = self.camera.cursor_to_world_position(self.window.cursor_pos)
                    world_delta = np.array([-(world_pos_old[0] - world_pos[0]), world_pos_old[1] - world_pos[1]]) / cfg.se_active_frame.pixel_size
                    self.crop_handles[0].move_crop_roi(world_delta[0], world_delta[1])

    def import_dataset(self, filename):
        try:
            # if os.path.isdir(filename):
            #     cfg.se_frames.append(SEFrame(filename, SPA=True))
            #     SegmentationEditor.set_active_dataset(cfg.se_frames[-1])
            _, ext = os.path.splitext(filename)
            if ext == ".mrc":
                cfg.se_frames.append(SEFrame(filename))
                SegmentationEditor.set_active_dataset(cfg.se_frames[-1])
                cfg.se_active_frame.requires_histogram_update = True
                self.parse_available_features()
            elif ext == cfg.filetype_segmentation:
                with open(filename, 'rb') as pickle_file:
                    seframe = pickle.load(pickle_file)
                    seframe.on_load()
                    seframe.slice_changed = False
                    cfg.se_frames.append(seframe)
                    SegmentationEditor.set_active_dataset(cfg.se_frames[-1])
                    # compatibility:
                    if not hasattr(seframe, 'crop'):
                        seframe.crop = False
                        seframe.crop_roi = [0, 0, seframe.width, seframe.height]
                    if not hasattr(seframe, 'includes_map'):
                        seframe.includes_map = False
                        seframe.map = None
                    if seframe.includes_map:
                        seframe.path = filename[:-len(cfg.filetype_segmentation)]+".mrc"  # 'virtual' file path, pointing at the location of the .scns file but ending with .mrc s.t. model output regex doesn't get messed up
                self.parse_available_features()
        except Exception as e:
            cfg.set_error(e, "Error importing dataset, see details below:")

    def gui_main(self):
        if True:
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
            imgui.push_style_color(imgui.COLOR_CHECK_MARK, 0.0, 0.0, 0.0, 1.0)
            imgui.push_style_color(imgui.COLOR_MENUBAR_BACKGROUND, *cfg.COLOUR_MAIN_MENU_BAR)
            imgui.push_style_color(imgui.COLOR_HEADER, *cfg.COLOUR_HEADER)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *cfg.COLOUR_HEADER_HOVERED)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *cfg.COLOUR_HEADER_ACTIVE)
            imgui.push_style_color(imgui.COLOR_TAB, *cfg.COLOUR_HEADER_ACTIVE)
            imgui.push_style_color(imgui.COLOR_TAB_ACTIVE, *cfg.COLOUR_HEADER)
            imgui.push_style_color(imgui.COLOR_TAB_HOVERED, *cfg.COLOUR_HEADER)
            imgui.push_style_color(imgui.COLOR_DRAG_DROP_TARGET, *cfg.COLOUR_DROP_TARGET)
            imgui.push_style_color(imgui.COLOR_SCROLLBAR_BACKGROUND, *cfg.COLOUR_WINDOW_BACKGROUND[:3], 0.0)
            imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, cfg.WINDOW_ROUNDING)

        def shared_gui():
            if imgui.collapsing_header("Datasets", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                panel_height = 120 if not SegmentationEditor.DATASETS_WINDOW_EXPANDED else SegmentationEditor.DATASETS_WINDOW_EXPANDED_HEIGHT
                imgui.begin_child("available_datasets", 0.0, panel_height, True, imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
                for s in cfg.se_frames:
                    imgui.push_id(f"se{s.uid}")

                    base_title = s.title.split(".")[0]
                    _change, _selected = imgui.selectable(base_title[-17:] + f" - {s.pixel_size * 10.0:.2f} A/pix ", cfg.se_active_frame == s)
                    if imgui.begin_popup_context_item("##datasetContext"):
                        if imgui.menu_item("Unlink dataset")[0]:
                            SegmentationEditor.pick_tab_index_datasets_segs = True
                            cfg.se_frames.remove(s)
                            if cfg.se_active_frame == s:
                                cfg.se_active_frame = None
                            self.parse_available_features()
                        if imgui.menu_item("Relink dataset")[0]:
                            selected_file = filedialog.askopenfilename(filetypes=[("mrcfile", ".mrc")])
                            s.path = selected_file
                            s.title = os.path.splitext(os.path.basename(s.path))[0]
                        if imgui.menu_item("Copy path to .mrc")[0]:
                            pyperclip.copy(s.path)
                        if s.overlay is not None and imgui.menu_item("Update overlay")[0]:
                            s.overlay.update()
                        if imgui.begin_menu("Generate binned version"):
                            path = s.path
                            for i in [2, 3, 4, 8]:
                                if imgui.menu_item(f"bin {i}")[0]:
                                    bpath = bin_mrc(path, i)
                                    self.import_dataset(bpath)
                            imgui.end_menu()
                        imgui.end_popup()
                    self.tooltip(f"{s.title}\nPixel size: {s.pixel_size * 10.0:.4f} Angstrom\nLocation: {s.path}")
                    if _change and _selected:
                        SegmentationEditor.set_active_dataset(s)
                        for model in cfg.se_models:
                            model.reset_textures()
                        self.trainset_apix = s.pixel_size * 10.0

                    for f in s.features:
                        imgui.same_line(spacing=1)
                        if imgui.color_button(f"##{f.uid}", f.colour[0], f.colour[1], f.colour[2], 1, imgui.COLOR_EDIT_NO_TOOLTIP, 6, 13):
                            s.active_feature = f

                    imgui.pop_id()
                imgui.end_child()
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_color(imgui.COLOR_CHECK_MARK, 0.0, 0.0, 0.0, 1.0)
                _, SegmentationEditor.DATASETS_WINDOW_EXPANDED = imgui.checkbox("expand", SegmentationEditor.DATASETS_WINDOW_EXPANDED)
                imgui.pop_style_var(3)
                imgui.pop_style_color(1)
            if imgui.collapsing_header("Filters", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                sef = cfg.se_active_frame
                if sef is not None:
                    _cw = imgui.get_content_region_available_width()
                    imgui.plot_histogram("##hist", sef.hist_vals,
                                         graph_size=(_cw, SegmentationEditor.INFO_HISTOGRAM_HEIGHT))
                    imgui.push_item_width(_cw)

                    imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                    imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                    imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                    imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                    imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_TEXT)

                    _minc, sef.contrast_lims[0] = imgui.slider_float("min", sef.contrast_lims[0], sef.hist_bins[0], sef.hist_bins[-1], format='min %.1f')
                    _maxc, sef.contrast_lims[1] = imgui.slider_float("max", sef.contrast_lims[1], sef.hist_bins[0], sef.hist_bins[-1], format='max %.1f')
                    if self.active_tab == "Pick":
                        _, SegmentationEditor.PICKING_FRAME_ALPHA = imgui.slider_float("alpha", SegmentationEditor.PICKING_FRAME_ALPHA, 0.0, 1.0, format="alpha %.1f")
                    if _minc or _maxc:
                        sef.autocontrast = False
                    imgui.pop_item_width()
                    _c, sef.invert = imgui.checkbox("inverted", sef.invert)
                    imgui.same_line(spacing=16)
                    _c, sef.autocontrast = imgui.checkbox("auto", sef.autocontrast)
                    if _c and sef.autocontrast:
                        sef.requires_histogram_update = True
                    imgui.same_line(spacing=16)
                    _c, sef.interpolate = imgui.checkbox("interpolate", sef.interpolate)
                    imgui.same_line(spacing=16)
                    if _c:
                        if sef.interpolate:
                            sef.texture.set_linear_interpolation()
                            SegmentationEditor.renderer.fbo1.texture.set_linear_interpolation()
                            SegmentationEditor.renderer.fbo2.texture.set_linear_interpolation()
                        else:
                            sef.texture.set_no_interpolation()
                            SegmentationEditor.renderer.fbo1.texture.set_no_interpolation()
                            SegmentationEditor.renderer.fbo2.texture.set_no_interpolation()
                    if self.active_tab in ["Segmentation", "Pick"]:
                        imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT_DISABLED)
                        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_TEXT_DISABLED)
                    _c, sef.crop = imgui.checkbox("crop", sef.crop)
                    if _c and not sef.crop:
                        sef.crop_roi = [0, 0, sef.width, sef.height]
                    if self.active_tab in ["Segmentation", "Pick"]:
                        imgui.pop_style_color(2)
                    imgui.separator()
                    fidx = 0
                    for ftr in self.filters:
                        fidx += 1
                        imgui.push_id(f"filter{fidx}")
                        cw = imgui.get_content_region_available_width()

                        # Filter type selection combo
                        imgui.set_next_item_width(cw - 60)
                        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (5, 2))
                        _c, ftrtype = imgui.combo("##filtertype", ftr.type, Filter.TYPES)
                        if _c:
                            self.filters[self.filters.index(ftr)] = Filter(ftrtype)
                        imgui.same_line()
                        _, ftr.enabled = imgui.checkbox("##enabled", ftr.enabled)
                        if _:
                            cfg.se_active_frame.requires_histogram_update = True
                        # Delete button
                        imgui.same_line()
                        if imgui.image_button(self.icon_close.renderer_id, 13, 13):
                            self.filters.remove(ftr)
                        imgui.pop_style_var(1)
                        # Parameter and strength sliders
                        imgui.push_item_width(cw)
                        if Filter.PARAMETER_NAME[ftr.type] is not None:
                            _c, ftr.param = imgui.slider_float("##param", ftr.param, 0.1, 10.0, format=f"{Filter.PARAMETER_NAME[ftr.type]}: {ftr.param:.1f}")
                            if _c:
                                ftr.fill_kernel()
                                cfg.se_active_frame.requires_histogram_update = True
                        _, ftr.strength = imgui.slider_float("##strength", ftr.strength, -1.0, 1.0, format=f"weight: {ftr.strength:.2f}")
                        if _:
                            cfg.se_active_frame.requires_histogram_update = True
                        imgui.pop_item_width()

                        imgui.pop_id()
                    imgui.set_next_item_width(imgui.get_content_region_available_width())
                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (5, 2))
                    _c, new_filter_type = imgui.combo("##filtertype", 0, ["Add filter"] + Filter.TYPES)
                    if _c and not new_filter_type == 0:
                        self.filters.append(Filter(new_filter_type - 1))
                        cfg.se_active_frame.requires_histogram_update = True

                    if cfg.se_active_frame.overlay is not None:
                        imgui.separator()
                        cw = imgui.get_content_region_available_width()
                        imgui.set_next_item_width(cw - 120)
                        _, SegmentationEditor.OVERLAY_ALPHA = imgui.slider_float("##alphaslider_se", SegmentationEditor.OVERLAY_ALPHA, 0.0, 1.0, format=f"overlay alpha = {SegmentationEditor.OVERLAY_ALPHA:.2f}") ## TODO 230808 werkt niet meer?
                        imgui.same_line()
                        imgui.set_next_item_width(110)
                        if self.active_tab != "Pick":
                            _, SegmentationEditor.OVERLAY_BLEND_MODE = imgui.combo("##overlayblending", SegmentationEditor.OVERLAY_BLEND_MODE, SegmentationEditor.BLEND_MODES_LIST)
                        else:
                            _, SegmentationEditor.OVERLAY_BLEND_MODE_3D = imgui.combo("##overlayblending", SegmentationEditor.OVERLAY_BLEND_MODE_3D,SegmentationEditor.BLEND_MODES_LIST_3D)
                            SegmentationEditor.VIEW_REQUIRES_UPDATE |= _
                            imgui.set_next_item_width(cw)
                            _, SegmentationEditor.OVERLAY_INTENSITY = imgui.slider_float("##inensityslider_se", SegmentationEditor.OVERLAY_INTENSITY, 0.0, 10.0, format=f"overlay intensity = {SegmentationEditor.OVERLAY_INTENSITY:.1f}")
                            SegmentationEditor.VIEW_REQUIRES_UPDATE |= _

                    imgui.pop_style_var(6)
                    imgui.pop_style_color(1)

        def segmentation_tab():

            def features_panel():
                if imgui.collapsing_header("Features", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    if cfg.se_active_frame is None:
                        return
                    features = cfg.se_active_frame.features
                    for f in features:
                        pop_active_colour = False
                        if cfg.se_active_frame.active_feature == f:
                            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *cfg.COLOUR_FRAME_ACTIVE)
                            pop_active_colour = True
                        imgui.begin_child(f"##feat_{f.uid}", 0.0, SegmentationEditor.FEATURE_PANEL_HEIGHT, True)
                        cw = imgui.get_content_region_available_width()

                        # Colour picker
                        _, f.colour = imgui.color_edit3(f.title, *f.colour[:3],
                                                        imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                        if _:
                            self.parse_available_features()
                        # Title
                        imgui.same_line()
                        imgui.set_next_item_width(cw - 26)
                        _, f.title = imgui.input_text("##title", f.title, 256, imgui.INPUT_TEXT_NO_HORIZONTAL_SCROLL | imgui.INPUT_TEXT_AUTO_SELECT_ALL)
                        if _:
                            self.parse_available_features()
                        self._gui_feature_title_context_menu(f)
                        # Alpha slider and brush size
                        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                        imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                        imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                        imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                        imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                        imgui.push_style_color(imgui.COLOR_CHECK_MARK, 0.0, 0.0, 0.0, 1.0)
                        imgui.set_next_item_width(cw - 40)
                        pxs = cfg.se_active_frame.pixel_size
                        _, f.brush_size = imgui.slider_float("brush", f.brush_size, 1.0, 25.0 / pxs,
                                                             format=f"{f.brush_size:.1f} px / {2 * f.brush_size * pxs:.1f} nm ")
                        f.brush_size = int(f.brush_size)
                        imgui.set_next_item_width(cw - 40)
                        _, f.alpha = imgui.slider_float("alpha", f.alpha, 0.0, 1.0, format="%.2f")
                        imgui.set_next_item_width(cw - 40)
                        _, f.box_size = imgui.slider_int("boxes", f.box_size, 8, 128, format=f"{f.box_size} pixel")
                        if _:
                            f.set_box_size(f.box_size)
                        # Show / fill checkboxes
                        _, show = imgui.checkbox("show", not f.hide)
                        f.hide = not show
                        imgui.same_line()
                        _, fill = imgui.checkbox("fill", not f.contour)
                        f.contour = not fill
                        imgui.same_line()
                        _, hide_boxes = imgui.checkbox("hide boxes", not f.show_boxes)
                        f.show_boxes = not hide_boxes
                        f.contour = not fill
                        imgui.same_line()
                        imgui.pop_style_color(1)
                        # delete feature button
                        imgui.same_line(position=cw - 20)
                        delete_feature = False
                        if imgui.image_button(self.icon_close.renderer_id, 13, 13):
                            delete_feature = True
                        imgui.same_line(position=cw - 20)

                        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *cfg.COLOUR_TRANSPARENT)
                        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *cfg.COLOUR_TRANSPARENT)
                        imgui.push_style_color(imgui.COLOR_HEADER, *cfg.COLOUR_TRANSPARENT)

                        if imgui.begin_menu("##asdc"):
                            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *cfg.COLOUR_HEADER_HOVERED)
                            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *cfg.COLOUR_HEADER_ACTIVE)
                            imgui.push_style_color(imgui.COLOR_HEADER, *cfg.COLOUR_HEADER)
                            n_boxes = sum([len(boxlist) for boxlist in f.boxes.values()])
                            imgui.text(f"Active slices ({len(f.edited_slices)}, {n_boxes} boxes)")

                            imgui.begin_child("active_slices", 200, SegmentationEditor.ACTIVE_SLICES_CHILD_HEIGHT, True)
                            cw = imgui.get_content_region_available_width()
                            for i in f.edited_slices:
                                imgui.push_id(f"{f.uid}{i}")

                                _, jumpto = imgui.selectable(f"Slice {i} ({len(f.boxes[i])} boxes)", f.current_slice == i, width=cw - 23)
                                if jumpto:
                                    f.parent.set_slice(i)
                                imgui.same_line(position=cw - 5)
                                if imgui.image_button(self.icon_close.renderer_id, 13, 13):
                                    f.remove_slice(i)
                                imgui.pop_id()
                            imgui.end_child()
                            imgui.text("Export:")
                            cw = imgui.get_content_region_available_width()
                            if imgui.button("coords", (cw - 15) / 3, 15):
                                f.save_particle_positions()
                            imgui.same_line(spacing=5)
                            if imgui.button("slice", (cw - 15) / 3, 15):
                                f.save_current_slice()
                            imgui.same_line(spacing=5)
                            if imgui.button("volume", (cw - 15) / 3, 15):
                                f.save_volume()
                            imgui.pop_style_color(3)
                            imgui.end_menu()

                        imgui.pop_style_color(3)

                        imgui.pop_style_var(5)
                        if pop_active_colour:
                            imgui.pop_style_color(1)

                        if delete_feature:
                            cfg.se_active_frame.features.remove(f)
                            if cfg.se_active_frame.active_feature == f:
                                cfg.se_active_frame.active_feature = None

                        if imgui.is_window_hovered() and imgui.is_mouse_clicked(0):
                            cfg.se_active_frame.active_feature = f


                        imgui.end_child()

                    # 'Add feature' button
                    cw = imgui.get_content_region_available_width()
                    imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)

                    imgui.new_line()
                    imgui.same_line(spacing=(cw - 120) / 2)
                    if imgui.button("Add feature", 120, 23):
                        cfg.se_active_frame.features.append(Segmentation(cfg.se_active_frame, f"Unnamed feature {len(cfg.se_active_frame.features)+1}"))
                        self.parse_available_features()
                    imgui.pop_style_var(1)


            features_panel()

        def models_tab():
            def calculate_number_of_boxes():
                self.trainset_num_boxes_positive = 0
                self.trainset_num_boxes_negative = 0
                for s in cfg.se_frames:
                    if s.sample:
                        for f in s.features:
                            if self.trainset_feature_selection[f.title] == 1:
                                self.trainset_num_boxes_positive += f.n_boxes
                            elif self.trainset_feature_selection[f.title] == -1:
                                self.trainset_num_boxes_negative += f.n_boxes


            if imgui.collapsing_header("Create a training set", None)[0]:
                self.show_trainset_boxes = True
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 2))

                imgui.text("Features of interest")
                imgui.begin_child("select_features", 0.0, 1 + len(self.all_feature_names) * 21, False, imgui.WINDOW_NO_SCROLLBAR)
                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_NEUTRAL_LIGHT)
                cw = imgui.get_content_region_available_width()
                imgui.push_item_width(cw)
                for fname in self.all_feature_names:
                    val = self.trainset_feature_selection[fname]
                    if val == 1:
                        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *cfg.COLOUR_POSITIVE)
                        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *cfg.COLOUR_POSITIVE)
                    elif val == 0:
                        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *cfg.COLOUR_NEUTRAL)
                        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *cfg.COLOUR_NEUTRAL)
                    elif val == -1:
                        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *cfg.COLOUR_NEGATIVE)
                        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *cfg.COLOUR_NEGATIVE)
                    _, self.trainset_feature_selection[fname] = imgui.slider_int(f"##{fname}", val, -1, 1, format=f"{fname}")
                    imgui.pop_style_color(2)
                imgui.pop_style_color(1)
                imgui.pop_style_var(1)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                imgui.end_child()

                imgui.text("Datasets to sample")
                imgui.push_style_color(imgui.COLOR_CHECK_MARK, 0.0, 0.0, 0.0, 1.0)
                imgui.begin_child("datasets_to_sample", 0.0, min([120, 10 + len(cfg.se_frames)*20]), True)
                for s in cfg.se_frames:
                    imgui.push_id(f"{s.uid}")
                    _, s.sample = imgui.checkbox(s.title, s.sample)
                    if _:
                        self.parse_available_features()
                    imgui.pop_id()
                imgui.end_child()
                imgui.pop_style_color()


                imgui.text("Set parameters")
                imgui.begin_child("params", 0.0, 80, True)
                imgui.push_item_width(cw - 53)
                _, self.trainset_boxsize = imgui.slider_int("boxes", self.trainset_boxsize, 8, 128, format=f"{self.trainset_boxsize} pixel")
                _, self.trainset_apix = imgui.slider_float("A/pix", self.trainset_apix, 1.0, 20.0, format=f"{self.trainset_apix:.2f}")
                imgui.pop_item_width()
                calculate_number_of_boxes()
                imgui.text(f"Positive samples: {self.trainset_num_boxes_positive}")
                imgui.text(f"Negative samples: {self.trainset_num_boxes_negative}")
                imgui.end_child()

                # progress bars
                for process in self.active_trainset_exports:
                    self._gui_background_process_progress_bar(process)
                    if process.progress == 1.0:
                        self.active_trainset_exports.remove(process)

                # 'Generate set' button
                if widgets.centred_button("Generate set", 120, 23):
                    self.launch_create_training_set()
                imgui.pop_style_var(5)
            else:
                self.show_trainset_boxes = False
            if imgui.collapsing_header("Models", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                for m in cfg.se_models:
                    imgui.push_id(f"SEModel_{m.uid}_main")
                    panel_height = 0
                    if m.active_tab == 0:
                        panel_height = SegmentationEditor.MODEL_PANEL_HEIGHT_TRAINING
                    elif m.active_tab == 1:
                        panel_height = SegmentationEditor.MODEL_PANEL_HEIGHT_PREDICTION
                    elif m.active_tab == 2:
                        panel_height = SegmentationEditor.MODEL_PANEL_HEIGHT_LOGIC + 57 * len(m.interactions) - 20 * (len(cfg.se_models) < 2)
                    panel_height += 10 if m.background_process_train is not None else 0
                    imgui.begin_child(f"SEModel_{m.uid}", 0.0, panel_height, True, imgui.WINDOW_NO_SCROLLBAR)
                    cw = imgui.get_content_region_available_width()

                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                    if m.background_process_train is None:
                        # delete button
                        if imgui.image_button(self.icon_close.renderer_id, 19, 19):
                            cfg.se_models.remove(m)
                            if cfg.se_active_model == m:
                                cfg.se_active_model = None
                            m.delete()
                    else:
                        if imgui.image_button(self.icon_stop.renderer_id, 19, 19):
                            m.background_process_train.stop()

                    imgui.pop_style_var(1)

                    imgui.same_line()
                    _, m.colour = imgui.color_edit3(m.title, *m.colour[:3], imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                    # Title
                    imgui.same_line()
                    imgui.set_next_item_width(cw - 55)
                    _, m.title = imgui.input_text("##title", m.title, 256, imgui.INPUT_TEXT_NO_HORIZONTAL_SCROLL | imgui.INPUT_TEXT_AUTO_SELECT_ALL)
                    self._gui_feature_title_context_menu(m)
                    # Model selection
                    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (5, 2))
                    imgui.align_text_to_frame_padding()

                    if m.compiled:
                        imgui.text(m.info)
                    else:
                        imgui.text("Model:")
                        imgui.same_line()
                        imgui.set_next_item_width(cw - 51)
                        _, m.model_enum = imgui.combo("##model_type", m.model_enum, SEModel.AVAILABLE_MODELS)
                    imgui.pop_style_var()

                    if imgui.begin_tab_bar("##tabs"):
                        if imgui.begin_tab_item(" Training ")[0]:
                            m.active_tab = 0
                            imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (3, 3))
                            # Train data selection
                            cw = imgui.get_content_region_available_width()
                            imgui.set_next_item_width(cw - 65)
                            _, m.train_data_path = imgui.input_text("##training_data", m.train_data_path, 256)
                            imgui.pop_style_var()
                            imgui.same_line()
                            if imgui.button("browse", 56, 19):
                                selected_file = filedialog.askopenfilename(filetypes=[("scNodes traindata", f"{cfg.filetype_traindata}")])
                                if selected_file is not None:
                                    m.train_data_path = selected_file

                            # Training parameters
                            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                            imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                            imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))

                            imgui.push_item_width((cw - 7) / 2)
                            _, m.epochs = imgui.slider_int("##epochs", m.epochs, 1, 50, f"{m.epochs} epoch"+("s" if m.epochs>1 else ""))
                            imgui.same_line()
                            excess_negative_label = f"{'' if m.excess_negative<0 else '+'}{m.excess_negative}%% negatives"
                            _, m.excess_negative = imgui.slider_int("##excessnegative", m.excess_negative, 0, 100, f"+{m.excess_negative}%% negatives")
                            _, m.batch_size = imgui.slider_int("##batchs", m.batch_size, 1, 128, f"{m.batch_size} batch size")
                            imgui.same_line()
                            _, m.n_copies = imgui.slider_int("##copies", m.n_copies, 1, 10, f"{m.n_copies} copies")
                            imgui.pop_item_width()
                            imgui.pop_style_var(1)
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 2))

                            # Load, save, train buttons.
                            block_buttons = False
                            if not m.background_process_train is None:
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_NEUTRAL_LIGHT)
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *cfg.COLOUR_NEUTRAL_LIGHT)
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_NEUTRAL_LIGHT)
                                imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_NEUTRAL)
                                block_buttons = True
                            if imgui.button("load", (cw - 16) / 3, 20):
                                if not block_buttons:
                                    model_path = filedialog.askopenfilename(filetypes=[("scNodes CNN", f"{cfg.filetype_semodel}")])
                                    if model_path != "":
                                        m.load(model_path)
                            imgui.same_line(spacing=8)
                            block_save_button = False
                            if m.model is None:
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_NEUTRAL_LIGHT)
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *cfg.COLOUR_NEUTRAL_LIGHT)
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_NEUTRAL_LIGHT)
                                imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_NEUTRAL)
                                block_save_button = True
                            if imgui.button("save", (cw - 16) / 3, 20):
                                if not block_buttons and not block_save_button:
                                    proposed_filename = f"{m.apix:.2f}_{m.box_size}_{m.loss:.4f}_{m.title}"
                                    model_path = filedialog.asksaveasfilename(filetypes=[("scNodes model", f"{cfg.filetype_semodel}")], initialfile=proposed_filename)
                                    if model_path != "":
                                        if model_path[-len(cfg.filetype_semodel):] != cfg.filetype_semodel:
                                            model_path += cfg.filetype_semodel
                                        m.save(model_path)
                            if block_save_button:
                                imgui.pop_style_color(4)
                            imgui.same_line(spacing=8)
                            if imgui.button("train", (cw - 16) / 3, 20):
                                if not block_buttons:
                                    m.train()
                            if block_buttons:
                                imgui.pop_style_color(4)

                            imgui.pop_style_var(5)
                            imgui.end_tab_item()

                        if imgui.begin_tab_item(" Prediction ")[0]:
                            m.active_tab = 1
                            # Checkboxes and sliders
                            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                            imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                            imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                            imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                            imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_TEXT)

                            imgui.push_item_width(imgui.get_content_region_available_width())
                            _, m.alpha = imgui.slider_float("##alpha", m.alpha, 0.0, 1.0, format=f"{m.alpha:.2f} alpha")
                            _, m.overlap = imgui.slider_float("##overlap", m.overlap, 0.0, 0.67, format=f"{m.overlap:.2f} overlap")
                            m.overlap = max(0.0, m.overlap)
                            _, m.threshold = imgui.slider_float("##thershold", m.threshold, 0.0, 1.0, format=f"{m.threshold:.2f} threshold")
                            imgui.pop_item_width()

                            _, m.active = imgui.checkbox("active   ", m.active)
                            if _ and m.active:
                                if cfg.se_active_frame is not None:
                                    cfg.se_active_frame.slice_changed = True
                            imgui.same_line()
                            _, m.blend = imgui.checkbox("blend   ", m.blend)
                            imgui.same_line()
                            _, m.show = imgui.checkbox("show     ", m.show)
                            imgui.same_line()
                            if imgui.button("save", 41, 14) and m.data is not None:
                                path = filedialog.asksaveasfilename(filetypes=[("mrcfile", ".mrc")], initialfile=os.path.basename(cfg.se_active_frame.path) + "_" + m.get_model_title() + f"_slice_{cfg.se_active_frame.current_slice}")
                                if path != "":
                                    if path[-4:] != ".mrc":
                                        path += ".mrc"
                                    with mrcfile.new(path, overwrite=True) as mrc:
                                        pxd = np.clip(m.data * 255, 0, 255).astype(
                                            np.uint8).squeeze()
                                        mrc.set_data(pxd)
                                        mrc.voxel_size = cfg.se_active_frame.pixel_size * 10.0
                            imgui.pop_style_var(5)
                            imgui.pop_style_color(1)
                            imgui.end_tab_item()

                        if imgui.begin_tab_item(" Interactions ")[0]:
                            m.active_tab = 2

                            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                            imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                            imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                            imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                            imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_TEXT)

                            imgui.text("Model competition:  ")
                            imgui.same_line()
                            _, m.emit = imgui.checkbox(" emit ", m.emit)
                            self.tooltip("When checked, this model emits prediction values, meaning it will affect\n"
                                         "absorbing models by nullifying their prediction wherever the emitting   \n"
                                         "model's prediction value is higher than that of an absorber.")
                            imgui.same_line()
                            _, m.absorb = imgui.checkbox(" absorb ", m.absorb)
                            self.tooltip("When checked, this model absorbs predictions by other models, meaning\n"
                                         "its output is nullified wherever there is any emitting model that \n"
                                         "predicts a higher value.")

                            # parse available models
                            available_partner_models = list()
                            available_partner_model_names = list()
                            for partner in cfg.se_models:
                                if partner != m:
                                    available_partner_models.append(partner)
                                    available_partner_model_names.append(partner.title)

                            # interaction GUI
                            cw = imgui.get_content_region_available_width()
                            for interaction in m.interactions:
                                delete_this_inter = False
                                if interaction.partner not in cfg.se_models:
                                    m.interactions.remove(interaction)
                                    continue
                                if interaction not in ModelInteraction.all:
                                    m.interactions.remove(interaction)
                                    continue
                                imgui.push_id(f"modelinteraction{interaction.uid}")

                                # model index
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, 1.0, 1.0, 1.0, 1.0)
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, 1.0, 1.0, 1.0, 1.0)
                                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, 1.0, 1.0, 1.0, 1.0)
                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (5, 1))
                                imgui.push_item_width(18)
                                m_idx = ModelInteraction.all.index(interaction)
                                _, new_idx = imgui.input_int("##interactionindex", m_idx+1, 0, 0)
                                if _:
                                    new_idx -= 1
                                    new_idx = min(max(0, new_idx), len(ModelInteraction.all) - 1)
                                    ModelInteraction.all[new_idx], ModelInteraction.all[m_idx] = ModelInteraction.all[m_idx], ModelInteraction.all[new_idx]
                                imgui.pop_style_color(3)
                                imgui.pop_style_var(1)

                                # select model to interact with
                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (8, 1))
                                imgui.set_next_item_width((cw - 18-9-15) // 2)
                                imgui.same_line(spacing=3)
                                _, interaction.type = imgui.combo("##type", interaction.type, ModelInteraction.TYPES)
                                if _:
                                    if cfg.se_active_frame is not None:
                                        cfg.se_active_frame.slice_changed = True
                                # select interaction type
                                p_idx = available_partner_model_names.index(interaction.partner.title)
                                imgui.same_line(spacing=3)
                                imgui.set_next_item_width((cw - 18-9-15) // 2)
                                imgui.push_style_color(imgui.COLOR_BUTTON, *interaction.partner.colour)
                                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *interaction.partner.colour)
                                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *interaction.partner.colour)
                                _, p_idx = imgui.combo("##partner_model", p_idx, available_partner_model_names)
                                if _:
                                    if cfg.se_active_frame is not None:
                                        cfg.se_active_frame.slice_changed = True
                                imgui.pop_style_color(3)
                                imgui.pop_style_var(1)
                                interaction.partner = available_partner_models[p_idx]

                                imgui.same_line(position = cw-7)
                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                                if imgui.image_button(self.icon_close.renderer_id, 15, 15):
                                    delete_this_inter = True
                                imgui.pop_style_var(1)
                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (3, 0))
                                imgui.set_next_item_width(cw)
                                _, interaction.radius = imgui.slider_float("##radius", interaction.radius, 0.0, 30.0,
                                                                           f"radius = {interaction.radius:.1f} nm")
                                if _:
                                    if cfg.se_active_frame is not None:
                                        cfg.se_active_frame.slice_changed = True
                                imgui.set_next_item_width(cw)
                                _, interaction.threshold = imgui.slider_float("##threshold", interaction.threshold, 0.0, 1.0,
                                                                           f"{interaction.partner.title} threshold = {interaction.threshold:.2f}")
                                if _:
                                    interaction.partner.threshold = interaction.threshold
                                    if cfg.se_active_frame is not None:
                                        cfg.se_active_frame.slice_changed = True
                                self.tooltip("Before applying interactions, the partner model's segmentation is thresholded at this value.")
                                imgui.pop_style_var(1)
                                imgui.separator()
                                imgui.pop_id()
                                if delete_this_inter:
                                    m.interactions.remove(interaction)
                                    ModelInteraction.all.remove(interaction)

                            # Add interaction
                            if len(available_partner_models) >= 1:
                                imgui.new_line()
                                imgui.same_line(spacing = (cw - 30) // 2)
                                if imgui.button("+", 30, 18):
                                    m.interactions.append(ModelInteraction(m, available_partner_models[0]))


                            imgui.pop_style_var(5)
                            imgui.pop_style_color(1)
                            imgui.end_tab_item()
                        imgui.end_tab_bar()

                    if m.background_process_train is not None:
                        self._gui_background_process_progress_bar(m.background_process_train)
                        if m.background_process_train.progress >= 1.0:
                            m.background_process_train = None
                    if m.background_process_apply is not None:
                        self._gui_background_process_progress_bar(m.background_process_apply)
                        if m.background_process_apply.progress >= 1.0:
                            m.background_process_apply = None

                    if imgui.is_window_hovered() and imgui.is_mouse_clicked(0):
                        cfg.se_active_model = m


                    imgui.end_child()
                    imgui.pop_id()
                cw = imgui.get_content_region_available_width()
                imgui.new_line()
                imgui.same_line(spacing=(cw - 120) / 2)
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                if imgui.button("Add model", 120, 23):
                    cfg.se_models.append(SEModel())
                imgui.pop_style_var(3)

        def export_tab():
            if imgui.collapsing_header("Export volumes", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                imgui.text("Models to include")
                n_available_models = sum([m.compiled for m in cfg.se_models])
                c_height = (1 if n_available_models == 0 else 9) + n_available_models * 21
                imgui.begin_child("models_included", 0.0, c_height, True)
                for m in cfg.se_models:
                    if not m.compiled:
                        continue
                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *m.colour)
                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *m.colour)
                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *m.colour)
                    imgui.push_style_color(imgui.COLOR_CHECK_MARK, 0.0, 0.0, 0.0, 1.0)
                    _, m.export = imgui.checkbox(m.title + " " + m.info_short, m.export)
                    if _:
                        m.active = m.export
                        m.show = m.export
                    imgui.pop_style_color(4)
                imgui.end_child()

                imgui.text("Datasets to process")

                imgui.same_line(spacing=80)
                _, SegmentationEditor.DATASETS_EXPORT_PANEL_EXPANDED = imgui.checkbox("expand",
                                                                                      SegmentationEditor.DATASETS_EXPORT_PANEL_EXPANDED)
                imgui.same_line(spacing=5)
                _, SegmentationEditor.DATASETS_EXPORT_SELECT_ALL = imgui.checkbox("all",
                                                                                  SegmentationEditor.DATASETS_EXPORT_SELECT_ALL)
                if _:
                    for s in cfg.se_frames:
                        s.export = SegmentationEditor.DATASETS_EXPORT_SELECT_ALL
                imgui.push_style_color(imgui.COLOR_CHECK_MARK, 0.0, 0.0, 0.0, 1.0)
                c_height = min([120, (1 if len(cfg.se_frames) == 0 else 9) + len(cfg.se_frames) * 21])
                if SegmentationEditor.DATASETS_EXPORT_PANEL_EXPANDED:
                    c_height = SegmentationEditor.DATASETS_EXPORT_PANEL_EXPANDED_HEIGHT
                imgui.begin_child("datasets_to_sample", 0.0, c_height, True)
                for s in cfg.se_frames:
                    imgui.push_id(f"{s.uid}")
                    if s == cfg.se_active_frame:
                        imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT_ACTIVE)
                        _, s.export = imgui.checkbox(s.title, s.export)
                        imgui.pop_style_color()
                    else:
                        _, s.export = imgui.checkbox(s.title, s.export)
                    if _:
                        self.parse_available_features()
                    imgui.pop_id()
                imgui.end_child()


                imgui.pop_style_color()

                imgui.text("Export settings")
                imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                imgui.begin_child("export_settings", 0.0, 58.0, True)
                _, self.export_dir = widgets.select_directory("browse", self.export_dir)
                imgui.set_next_item_width(imgui.get_content_region_available_width())
                _, self.export_limit_range = imgui.checkbox(" limit range ", self.export_limit_range)
                imgui.same_line(spacing=62)
                _, self.export_overlays = imgui.checkbox(" export overlays", self.export_overlays)
                imgui.end_child()


                if widgets.centred_button("Start export", 120, 23):
                    self.launch_export_volumes()

                # export progress:
                if self.queued_exports:
                    imgui.spacing()
                    imgui.text(f"Exporting tomograms - {len(self.queued_exports)} jobs remaining.")
                    cancel = self._gui_background_process_progress_bar(self.queued_exports[0].process, (*self.queued_exports[0].colour, 1.0), cancellable=True)
                    if cancel:
                        self.queued_exports[0].stop()
                        self.queued_exports.pop(0)

                imgui.pop_style_var(4)

        def picking_tab():
            if imgui.collapsing_header("Volumes", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                imgui.push_style_color(imgui.COLOR_CHECK_MARK, 0.0, 0.0, 0.0)

                # list the segmentations found for the currently selected dataset.
                def update_picking_tab_for_new_active_frame():
                    # delete old surface models
                    for s in cfg.se_surface_models:
                        s.delete()
                    cfg.se_surface_models = list()
                    # index new surface models - find corresponding segmentation .mrc's
                    if cfg.se_active_frame is None:
                        self.pick_box_va.update(VertexBuffer([0.0, 0.0]), IndexBuffer([]))
                        self.pick_box_quad_va.update(VertexBuffer([0.0, 0.0]), IndexBuffer([]))
                        return
                    se_frame = cfg.se_active_frame
                    SegmentationEditor.pick_tab_index_datasets_segs = False
                    # find segmentations in the selected dataset's folder.

                    if self.pick_same_folder:
                        files = glob.glob(os.path.join(os.path.dirname(se_frame.path), se_frame.title + "_*.mrc"))
                        print(f"Filename template: {os.path.dirname(se_frame.path)}, {se_frame.title}, _*.mrc")
                    else:
                        files = glob.glob(os.path.join(self.seg_folder, se_frame.title + "_*.mrc"))
                        print(f"Filename template: {self.seg_folder}, {se_frame.title}, _*.mrc")
                    for f in sorted(files):
                        if not 'overlay' in f:
                            cfg.se_surface_models.append(SurfaceModel(f, se_frame.pixel_size))

                    # set the size of the volume spanning box
                    w, h, d = se_frame.width / 2, se_frame.height / 2, se_frame.n_slices / 2
                    w *= se_frame.pixel_size
                    h *= se_frame.pixel_size
                    d *= se_frame.pixel_size
                    vertices = [-w, h, d,
                                w, h, d,
                                w, -h, d,
                                -w, -h, d,
                                -w, h, -d,
                                w, h, -d,
                                w, -h, -d,
                                -w, -h, -d]
                    line_indices = [0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 1, 5, 2, 6, 3, 7, 4, 5, 5, 6, 6, 7, 7, 4]
                    quad_indices = [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 0, 4, 7, 7, 3, 0, 5, 1, 2, 2, 6, 5, 4, 0, 1, 1, 5, 4, 3, 7, 6, 6, 2, 3]
                    self.pick_box_va.update(VertexBuffer(vertices), IndexBuffer(line_indices))
                    self.pick_box_quad_va.update(VertexBuffer(vertices), IndexBuffer(quad_indices))


                if SegmentationEditor.pick_tab_index_datasets_segs:
                    update_picking_tab_for_new_active_frame()

                for s in cfg.se_surface_models:
                    req_gen_models = False
                    s.on_update()
                    #imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *s.colour, 0.0 if s.hide else 0.2)
                    imgui.begin_child(f"{s.title}_surfm", 0.0, 82, True)
                    cw = imgui.get_content_region_available_width()
                    _, s.colour = imgui.color_edit3(s.title, *s.colour[:3], imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)

                    imgui.same_line()
                    imgui.text(f"{s.title} ({s.size_mb:.1f} mb)")
                    imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                    imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                    imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)

                    imgui.same_line(position = cw - 40)
                    _, s.hide = imgui.checkbox("hide  ", s.hide)
                    # progress bar
                    if s.process is not None:
                        if s.process.progress == 1.0:
                            s.process = None

                    imgui.push_item_width(cw)
                    original_level = s.level
                    _, s.level = imgui.slider_int("##level", s.level, 0, 256, f"level = {s.level}")
                    if s.process is not None:
                        s.level = original_level
                        _ = False
                    if _ and original_level != s.level:
                        req_gen_models = True

                    _, s.dust = imgui.slider_float("##dust", s.dust, 10.0, 1000000.0, f"dust < {s.dust:.1f} nm³", imgui.SLIDER_FLAGS_LOGARITHMIC)
                    s.hide_dust()
                    imgui.pop_item_width()
                    imgui.push_item_width((cw - 7) / 2)
                    _, s.alpha = imgui.slider_float("##alpha", s.alpha, 0, 1.0, f"alpha = {s.alpha:.1f}")
                    imgui.same_line()
                    original_bin = s.bin
                    _, s.bin = imgui.slider_int("##bin", s.bin, 1, 8, f"bin {s.bin}")
                    if _ and original_bin != s.bin:
                        req_gen_models = True


                    if req_gen_models:
                        s.generate_model()

                    imgui.pop_item_width()
                    imgui.pop_style_var(3)
                    imgui.end_child()

                imgui.pop_style_var(3)
                imgui.pop_style_color(1)

            # if imgui.collapsing_header("Extract coordinates", None)[0]:
            #     imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
            #     imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
            #     imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
            #     imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 10)
            #     imgui.text("Detection settings")
            #     imgui.begin_child("extract_feature", 0.0, 80 if len(cfg.se_surface_models) > 0 else 1, True)
            #     if len(cfg.se_surface_models) > 0:
            #         cw = imgui.get_content_region_available_width()
            #         available_features = [s.title for s in cfg.se_surface_models]
            #         selected_feature_idx = 0 if SegmentationEditor.EXTRACT_SELECTED_FEATURE_TITLE not in available_features else available_features.index(SegmentationEditor.EXTRACT_SELECTED_FEATURE_TITLE)
            #         selected_surface_model = cfg.se_surface_models[selected_feature_idx]
            #
            #         imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (10, 0))
            #         imgui.push_style_color(imgui.COLOR_BUTTON, *selected_surface_model.colour, 1.0)
            #         imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *selected_surface_model.colour, 1.0)
            #         imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *selected_surface_model.colour, 1.0)
            #         imgui.push_item_width(cw)
            #         _, selected_feature_idx = imgui.combo("##sel_ext_ftr", selected_feature_idx, available_features)
            #         imgui.pop_style_var(1)
            #         imgui.pop_style_color(3)
            #         _, SegmentationEditor.EXTRACT_THRESHOLD = imgui.slider_int("##ext_thr", SegmentationEditor.EXTRACT_THRESHOLD, 0, 255, format=f"threshold = {SegmentationEditor.EXTRACT_THRESHOLD}")
            #         _, SegmentationEditor.EXTRACT_MIN_WEIGHT = imgui.slider_int("##ext_mw", SegmentationEditor.EXTRACT_MIN_WEIGHT, 1000, 100000, format=f"min weight = {SegmentationEditor.EXTRACT_MIN_WEIGHT}", flags=imgui.SLIDER_FLAGS_LOGARITHMIC)
            #         _, SegmentationEditor.EXTRACT_MIN_SPACING = imgui.slider_int("##ext_ms", SegmentationEditor.EXTRACT_MIN_SPACING, 10, 300, format=f"min spacing = {SegmentationEditor.EXTRACT_MIN_SPACING} A")
            #     imgui.end_child()
            #
            #     imgui.text("Datasets to process")
            #     imgui.same_line(spacing=80)
            #     _, SegmentationEditor.DATASETS_PICK_PANEL_EXPANDED = imgui.checkbox("expand", SegmentationEditor.DATASETS_PICK_PANEL_EXPANDED)
            #     imgui.same_line(spacing=5)
            #     _, SegmentationEditor.DATASETS_PICK_SELECT_ALL = imgui.checkbox("all", SegmentationEditor.DATASETS_PICK_SELECT_ALL)
            #     if _:
            #         for s in cfg.se_frames:
            #             s.pick = SegmentationEditor.DATASETS_PICK_SELECT_ALL
            #     imgui.push_style_color(imgui.COLOR_CHECK_MARK, 0.0, 0.0, 0.0, 1.0)
            #     c_height = min([120, (1 if len(cfg.se_frames) == 0 else 9) + len(cfg.se_frames) * 21])
            #     if SegmentationEditor.DATASETS_PICK_PANEL_EXPANDED:
            #         c_height = SegmentationEditor.DATASETS_PICK_PANEL_EXPANDED_HEIGHT
            #
            #     imgui.begin_child("datasets_to_sample", 0.0, c_height, True)
            #     for s in cfg.se_frames:
            #         imgui.push_id(f"{s.uid}")
            #         if s == cfg.se_active_frame:
            #             imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT_ACTIVE)
            #             _, s.pick = imgui.checkbox(s.title, s.pick)
            #             imgui.pop_style_color()
            #         else:
            #             _, s.pick = imgui.checkbox(s.title, s.pick)
            #         if _:
            #             self.parse_available_features()
            #         imgui.pop_id()
            #     imgui.end_child()
            #     imgui.pop_style_color(1)
            #     ## buttons
            #     cw = imgui.get_content_region_available_width()
            #     w = 110
            #     imgui.new_line()
            #     imgui.same_line(spacing=(cw - 2 * w) / 3)
            #     if imgui.button("Trial", w, 20):
            #         selected_surface_model.find_coordinates(SegmentationEditor.EXTRACT_THRESHOLD, SegmentationEditor.EXTRACT_MIN_WEIGHT, SegmentationEditor.EXTRACT_MIN_SPACING)
            #     imgui.same_line(spacing=(cw - 2 * w) / 3)
            #     if imgui.button("Export all", w, 20):
            #         self.launch_export_coordinates(selected_surface_model.title)
            #     if selected_surface_model.coordinates is not None:
            #         print("TODO: render particle ROIs")  ## TODO
            #
            #     ## extract progress bar & cancel option:
            #     if self.queued_extracts:
            #         imgui.spacing()
            #         imgui.text(f"Exporting coordinates - {len(self.queued_extracts)} jobs remaining.")
            #         cancel = self._gui_background_process_progress_bar(self.queued_extracts[0].process, cfg.COLOUR_POSITIVE, cancellable=True)
            #         if cancel:
            #             self.queued_extracts[0].stop()
            #             self.queued_extracts.pop(0)
            #
            #     imgui.pop_style_var(4)

            SegmentationEditor.LIGHT_SPOT.compute_vec(dyaw=-self.camera3d.yaw)
            if imgui.collapsing_header("Graphics settings", None)[0]:
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (2, 2))
                cw = imgui.get_content_region_available_width()
                imgui.align_text_to_frame_padding()
                imgui.text("Volume render style:")
                imgui.same_line()
                imgui.push_item_width(cw - 145)
                _, SegmentationEditor.SELECTED_RENDER_STYLE = imgui.combo("##Graphics style", SegmentationEditor.SELECTED_RENDER_STYLE, SegmentationEditor.RENDER_STYLES)
                imgui.pop_item_width()
                _, SegmentationEditor.RENDER_CLEAR_COLOUR = imgui.color_edit3("##clrclr", *SegmentationEditor.RENDER_CLEAR_COLOUR[:3], imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                imgui.same_line()
                imgui.align_text_to_frame_padding()
                imgui.text(" Background colour      ")
                imgui.same_line()
                _, SegmentationEditor.LIGHT_SPOT.colour = imgui.color_edit3("##lightclr", *SegmentationEditor.LIGHT_SPOT.colour[:3], imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                imgui.same_line()
                imgui.align_text_to_frame_padding()
                imgui.text(" Light colour")

                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 10)
                imgui.push_item_width(cw)
                _, SegmentationEditor.LIGHT_AMBIENT_STRENGTH = imgui.slider_float("##ambient lighting", SegmentationEditor.LIGHT_AMBIENT_STRENGTH, 0.0, 1.0, f"ambient strength = %.2f")
                _, SegmentationEditor.LIGHT_SPOT.strength = imgui.slider_float("##spot lighting", SegmentationEditor.LIGHT_SPOT.strength, 0.0, 1.0, f"spot strength = %.2f")
                _yaw, SegmentationEditor.LIGHT_SPOT.yaw = imgui.drag_float("##yaw", SegmentationEditor.LIGHT_SPOT.yaw, 0.5, format=f"spot yaw = %.1f")
                _pitch, SegmentationEditor.LIGHT_SPOT.pitch = imgui.slider_float("##pitch", SegmentationEditor.LIGHT_SPOT.pitch, -90.0, 90.0, format=f"spot pitch = %.1f")
                imgui.pop_item_width()

                imgui.push_style_color(imgui.COLOR_CHECK_MARK, 0.0, 0.0, 0.0)
                _, SegmentationEditor.RENDER_BOX = imgui.checkbox(" render bounding box     ",
                                                                  SegmentationEditor.RENDER_BOX)
                imgui.same_line()
                _render_frame = SegmentationEditor.PICKING_FRAME_ALPHA != 0.0
                _r, _render_frame = imgui.checkbox(" render frame", _render_frame)
                if _r:
                    SegmentationEditor.PICKING_FRAME_ALPHA = float(_render_frame)
                imgui.pop_style_var(4)
                imgui.pop_style_color(1)

                if widgets.centred_button('Recompile Shaders', 150, 20):
                    self.renderer.recompile_shaders()

                imgui.pop_style_var(1)

            if imgui.collapsing_header("Export 3D scene", None)[0]:
                cw = imgui.get_content_region_available_width()
                spacing = 10
                s = (cw - 2 * spacing * 2 - 3) / 3
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 100)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *cfg.COLOUR_FRAME_DARK)
                imgui.push_style_color(imgui.COLOR_BORDER, *cfg.COLOUR_FRAME_DARK)
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_FRAME_EXTRA_DARK)
                if imgui.image_button(self.icon_obj.renderer_id, s, s):
                    pass
                imgui.same_line(spacing=spacing)
                if imgui.image_button(self.icon_blender.renderer_id, s, s):
                    pass
                imgui.same_line(spacing=spacing)
                if imgui.image_button(self.icon_chimerax.renderer_id, s, s):
                    pass
                imgui.pop_style_var(2)
                imgui.pop_style_color(3)
                # Open in Blender
                # Save to .obj
                # Open in ChimeraX


        def menu_bar():
            imgui.push_style_color(imgui.COLOR_MENUBAR_BACKGROUND, *cfg.COLOUR_MAIN_MENU_BAR)
            imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_MAIN_MENU_BAR_TEXT)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *cfg.COLOUR_MAIN_MENU_BAR_HILIGHT)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *cfg.COLOUR_MAIN_MENU_BAR_HILIGHT)
            imgui.push_style_color(imgui.COLOR_HEADER, *cfg.COLOUR_MAIN_MENU_BAR_HILIGHT)
            imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *cfg.COLOUR_MENU_WINDOW_BACKGROUND)
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (2.0, 2.0))

            if imgui.core.begin_main_menu_bar():
                if imgui.begin_menu("File"):
                    if imgui.menu_item("Import dataset")[0]:
                        try:
                            filename = filedialog.askopenfilename(filetypes=[("scNodes segmentable", f".mrc {cfg.filetype_segmentation}")])
                            if filename != '':
                                self.import_dataset(filename)
                        except Exception as e:
                            cfg.set_error(e, "Could not import dataset, see details below.")
                    if imgui.menu_item("Import model group")[0]:
                        try:
                            filename = filedialog.askopenfilename(filetypes=[("scNodes model group", cfg.filetype_semodel_group)])
                            if filename != '':
                                SegmentationEditor.load_model_group(filename)
                        except Exception as e:
                            cfg.set_error(e, "Could not import model group, see details below.")
                    # if imgui.menu_item("Import SPA dataset")[0]:
                    #     try:
                    #         dirname = filedialog.askdirectory()
                    #         if dirname != '':
                    #             self.import_dataset(dirname)
                    #     except Exception as e:
                    #         cfg.set_error(e, "Could not import SPA dataset, see details below")
                    if imgui.menu_item("Save dataset")[0]:
                        try:
                            filename = filedialog.asksaveasfilename(filetypes=[("scNodes segmentation", f"{cfg.filetype_segmentation}")], initialfile = os.path.basename(cfg.se_active_frame.path)[:-4])
                            if filename != '':
                                if filename[-len(cfg.filetype_segmentation):] != cfg.filetype_segmentation:
                                    filename += cfg.filetype_segmentation
                                with open(filename, 'wb') as pickle_file:
                                    pickle.dump(cfg.se_active_frame, pickle_file)
                        except Exception as e:
                            cfg.set_error(e, "Could not save dataset, see details below.")
                    if imgui.menu_item("Save dataset w. map")[0]:
                        try:
                            filename = filedialog.asksaveasfilename(filetypes=[("scNodes segmentation", f"{cfg.filetype_segmentation}")], initialfile = os.path.basename(cfg.se_active_frame.path)[:-4])
                            if filename != '':
                                if filename[-len(cfg.filetype_segmentation):] != cfg.filetype_segmentation:
                                    filename += cfg.filetype_segmentation
                                cfg.se_active_frame.include_map()
                                with open(filename, 'wb') as pickle_file:
                                    pickle.dump(cfg.se_active_frame, pickle_file)
                        except Exception as e:
                            cfg.set_error(e, "Could not save dataset (including map), see details below.")
                    if imgui.menu_item("Save model group")[0]:
                        try:
                            filename = filedialog.asksaveasfilename(filetypes=[("scNodes model group", f"{cfg.filetype_semodel_group}")])
                            if filename != '':
                                if filename[-len(cfg.filetype_semodel_group):] != cfg.filetype_semodel_group:
                                    filename += cfg.filetype_semodel_group
                                SegmentationEditor.save_model_group(filename)
                        except Exception as e:
                            cfg.set_error(e, "Could not save model group, see details below.")

                    imgui.end_menu()
                if imgui.begin_menu("Editor"):
                    for i in range(len(cfg.editors)):
                        select, _ = imgui.menu_item(cfg.editors[i], None, False)
                        if select:
                            cfg.active_editor = i
                    imgui.end_menu()


                imgui.end_main_menu_bar()

            imgui.pop_style_color(6)
            imgui.pop_style_var(1)

        def slicer_window():
            if cfg.se_active_frame is None:
                return

            imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, cfg.CE_WIDGET_ROUNDING)
            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, cfg.CE_WIDGET_ROUNDING)
            imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1.0)

            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_PANEL_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *cfg.COLOUR_PANEL_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_PANEL_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *cfg.COLOUR_FRAME_EXTRA_DARK[0:3], 0.8)
            imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *cfg.COLOUR_FRAME_EXTRA_DARK[0:3], 0.8)
            imgui.set_next_window_size(SegmentationEditor.SLICER_WINDOW_WIDTH, 0.0)
            window_x_pos = SegmentationEditor.MAIN_WINDOW_WIDTH + (self.window.width - SegmentationEditor.MAIN_WINDOW_WIDTH - SegmentationEditor.SLICER_WINDOW_WIDTH) / 2

            export_mode = self.export_limit_range and self.active_tab == "Export"
            vertical_offset = self.window.height - SegmentationEditor.SLICER_WINDOW_VERTICAL_OFFSET - (40 if export_mode else 23)
            imgui.set_next_window_position(window_x_pos, vertical_offset)
            imgui.begin("##slicer", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_BACKGROUND)

            cw = imgui.get_content_region_available_width()
            imgui.push_item_width(cw)

            frame = cfg.se_active_frame
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_FRAME_BACKGROUND[0:3], 0.7)
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_FRAME_BACKGROUND[0:3], 0.7)
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *cfg.COLOUR_FRAME_BACKGROUND[0:3], 0.7)
            imgui.push_style_color(imgui.COLOR_BORDER, 0.3, 0.3, 0.3, 1.0)

            if export_mode:
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                _, frame.export_top = imgui.slider_int("##export_top", frame.export_top, 1, frame.n_slices, format=f"export stop {frame.export_top + 1}")
                imgui.pop_style_var(1)
                if _:
                    frame.set_slice(frame.export_top)
                    frame.export_bottom = min([frame.export_bottom, frame.export_top - 1])
                    frame.export_bottom = max([frame.export_bottom, 1])
                origin = imgui.get_window_position()
                y = imgui.get_cursor_screen_pos()[1]
                draw_list = imgui.get_background_draw_list()
                left = 8 + origin[0] + cw * frame.export_bottom / frame.n_slices
                right = 8 + origin[0] + cw * frame.export_top / frame.n_slices
                draw_list.add_rect_filled(left, y, right, y + 18, imgui.get_color_u32_rgba(*cfg.COLOUR_POSITIVE), 10)

            _, requested_slice = imgui.slider_int("##slicer_slider", frame.current_slice, 0, frame.n_slices, format=f"slice {1+frame.current_slice}/{frame.n_slices}")
            if _:
                frame.set_slice(requested_slice)

            if export_mode:
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                _, frame.export_bottom = imgui.slider_int("##export_bottom", frame.export_bottom, 0, frame.n_slices - 1, format=f"export start {frame.export_bottom + 1}")
                imgui.pop_style_var(1)
                if _:
                    frame.set_slice(frame.export_bottom)
                    frame.export_top = max([frame.export_top, frame.export_bottom + 1])
                    frame.export_top = min([frame.export_top, frame.n_slices])
            imgui.pop_style_color(4)
            imgui.pop_item_width()
            imgui.end()

            imgui.pop_style_var(3)
            imgui.pop_style_color(5)

        # START GUI:
        # Menu bar
        menu_bar()
        # Render the currently active frame

        if cfg.se_active_frame is not None:
            if self.active_tab != "Pick":
                pxd = SegmentationEditor.renderer.render_filtered_frame(cfg.se_active_frame, self.camera, self.window, self.filters, emphasize_roi=self.active_tab != "Segmentation") ## todo: save the output pxd to the corresponding frame, and render this pxd in subsequent app frames where the data is not changed instead of applying the filters on GPU every frame - which it turns out is fairly expensive
                if pxd is not None:
                    cfg.se_active_frame.compute_histogram(pxd)
                    if cfg.se_active_frame.autocontrast:
                        cfg.se_active_frame.compute_autocontrast(None, pxd)
                    cfg.se_active_frame.requires_histogram_update = False
            if self.active_tab == "Segmentation":
                SegmentationEditor.renderer.render_segmentations(cfg.se_active_frame, self.camera)

                # render drawing ROI indicator
                active_feature = cfg.se_active_frame.active_feature
                if active_feature is not None:
                    radius = active_feature.brush_size * active_feature.parent.pixel_size
                    world_position = self.camera.cursor_to_world_position(self.window.cursor_pos)
                    if not imgui.is_key_down(glfw.KEY_LEFT_SHIFT):
                        SegmentationEditor.renderer.add_circle(world_position, radius, active_feature.colour)
                    else:
                        SegmentationEditor.renderer.add_square(world_position, active_feature.box_size_nm, active_feature.colour)
                for f in cfg.se_active_frame.features:
                    frame_xy = f.parent.transform.translation
                    if f.show_boxes and not f.hide and f.current_slice in f.boxes:
                        for box in f.boxes[f.current_slice]:
                            box_x_pos = frame_xy[0] + (box[0] - f.parent.width / 2) * f.parent.pixel_size
                            box_y_pos = frame_xy[1] + (box[1] - f.parent.height / 2) * f.parent.pixel_size
                            SegmentationEditor.renderer.add_square((box_x_pos, box_y_pos), f.box_size_nm, f.colour)
            if self.active_tab in ["Models", "Export"]:
                SegmentationEditor.renderer.render_models(cfg.se_active_frame, self.camera)
            if self.active_tab == "Models" and self.show_trainset_boxes:
                for f in cfg.se_active_frame.features:
                    frame_xy = f.parent.transform.translation
                    if f.current_slice in f.boxes and f.title in self.trainset_feature_selection:
                        sval = self.trainset_feature_selection[f.title]
                        if sval == 0.0:
                            continue
                        clr = cfg.COLOUR_POSITIVE if sval == 1.0 else cfg.COLOUR_NEGATIVE
                        for box in f.boxes[f.current_slice]:
                            box_x_pos = frame_xy[0] + (box[0] - f.parent.width / 2) * f.parent.pixel_size
                            box_y_pos = frame_xy[1] + (box[1] - f.parent.height / 2) * f.parent.pixel_size
                            box_size = self.trainset_apix * self.trainset_boxsize / 10.0
                            SegmentationEditor.renderer.add_square((box_x_pos, box_y_pos), box_size, clr)
            if self.active_tab != "Pick":
                overlay_blend_mode = SegmentationEditor.BLEND_MODES[SegmentationEditor.BLEND_MODES_LIST[SegmentationEditor.OVERLAY_BLEND_MODE]]
                SegmentationEditor.renderer.render_overlay(cfg.se_active_frame, self.camera, overlay_blend_mode, SegmentationEditor.OVERLAY_ALPHA)
            else:
                if SegmentationEditor.RENDER_BOX:
                    # Render 3D box around the volume
                    self.renderer.render_line_va(self.pick_box_va, self.camera3d)

                # Render lines along edges of frame
                self.renderer.render_frame_border(cfg.se_active_frame, self.camera3d)

                # Render surface models in 3D
                if not imgui.is_key_down(glfw.KEY_SPACE):
                    self.renderer.render_surface_models(cfg.se_surface_models, self.camera3d, SegmentationEditor.LIGHT_AMBIENT_STRENGTH, SegmentationEditor.LIGHT_SPOT)

                # Render the frame
                pxd = SegmentationEditor.renderer.render_filtered_frame(cfg.se_active_frame, self.camera, self.window, self.filters, camera3d=self.camera3d)
                if pxd is not None:
                    cfg.se_active_frame.compute_histogram(pxd)
                    if cfg.se_active_frame.autocontrast:
                        cfg.se_active_frame.compute_autocontrast(None, pxd)
                    cfg.se_active_frame.requires_histogram_update = False

                # Render overlay, if possible:
                if cfg.se_active_frame.overlay is not None:
                    if SegmentationEditor.VIEW_REQUIRES_UPDATE:
                        self.renderer.ray_trace_overlay((self.window.width, self.window.height), cfg.se_active_frame, self.camera3d, self.pick_box_quad_va)
                    self.renderer.render_overlay_3d(SegmentationEditor.OVERLAY_ALPHA, SegmentationEditor.OVERLAY_INTENSITY)
        # MAIN WINDOW
        imgui.set_next_window_position(0, 17, imgui.ONCE)
        imgui.set_next_window_size(SegmentationEditor.MAIN_WINDOW_WIDTH, self.window.height - 17)

        imgui.begin("##se_main", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        shared_gui()
        imgui.spacing()
        imgui.spacing()


        if imgui.begin_tab_bar("##tabs"):
            self.window.clear_color = cfg.COLOUR_WINDOW_BACKGROUND
            if imgui.begin_tab_item(" Segmentation ")[0]:
                segmentation_tab()
                self.active_tab = "Segmentation"
                imgui.end_tab_item()
            if imgui.begin_tab_item(" Models ")[0]:
                if self.active_tab != "Models":
                    self.parse_available_features()
                self.active_tab = "Models"
                models_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item(" Export ")[0]:
                self.active_tab = "Export"
                export_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item(" Render ")[0]:
                self.window.clear_color = [*SegmentationEditor.RENDER_CLEAR_COLOUR, 1.0]
                self.active_tab = "Pick"
                picking_tab()
                imgui.end_tab_item()
            imgui.end_tab_bar()

        imgui.end()

        slicer_window()
        self._warning_window()
        imgui.pop_style_color(32)
        imgui.pop_style_var(1)

    def _warning_window(self):
        def ww_context_menu():
            imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *cfg.COLOUR_MENU_WINDOW_BACKGROUND)
            if imgui.begin_popup_context_window():
                cfg.error_window_active = True
                copy_error, _ = imgui.menu_item("Copy to clipboard")
                if copy_error:
                    pyperclip.copy(cfg.error_msg)
                copy_path_to_log, _ = imgui.menu_item("Copy path to scNodes.log")
                if copy_path_to_log:
                    pyperclip.copy(os.path.abspath(cfg.logpath))
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
            window_vertical_offset = 0 if cfg.error_window_active else cfg.ERROR_WINDOW_HEIGHT - 19
            imgui.set_next_window_position(0, self.window.height - cfg.ERROR_WINDOW_HEIGHT + window_vertical_offset)
            _, stay_open = imgui.begin("Notification", True, imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE)
            if imgui.is_window_focused():
                cfg.error_window_active = True
                if self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.PRESS):
                    cfg.error_new = False
            else:
                cfg.error_window_active = False
            if not cfg.error_logged:
                cfg.write_to_log(cfg.error_msg)
                cfg.error_logged = True
            imgui.text(cfg.error_msg)
            ww_context_menu()
            imgui.end()
            if not stay_open:
                cfg.error_msg = None
                cfg.error_new = True
            imgui.pop_style_color(5)
            imgui.pop_style_var(1)

    def parse_available_features(self):
        # upon opening Models tab.
        self.all_feature_names = list()
        self.feature_colour_dict = dict()
        for sef in cfg.se_frames:
            for ftr in sef.features:
                if sef.sample:
                    if ftr.title not in self.all_feature_names:
                        self.all_feature_names.append(ftr.title)
                    if ftr.title not in self.trainset_feature_selection:
                        self.trainset_feature_selection[ftr.title] = 0.0
                if ftr.title not in self.feature_colour_dict:
                    self.feature_colour_dict[ftr.title] = ftr.colour
        to_pop = list()
        for key in self.trainset_feature_selection.keys():
            if key not in self.all_feature_names:
                to_pop.append(key)
        for key in to_pop:
            self.trainset_feature_selection.pop(key)

    def _gui_background_process_progress_bar(self, process, colour=cfg.COLOUR_POSITIVE, cancellable=False, height=None):
        height = SegmentationEditor.PROGRESS_BAR_HEIGHT if height is None else height
        cw = imgui.get_content_region_available_width()
        cancel_button_w = SegmentationEditor.PROGRESS_BAR_HEIGHT + 5 if cancellable else 0
        origin = imgui.get_window_position()
        y = imgui.get_cursor_screen_pos()[1]
        drawlist = imgui.get_window_draw_list()
        drawlist.add_rect_filled(8 + origin[0], y,
                                 8 + origin[0] + max(0, cw - cancel_button_w),
                                 y + height,
                                 imgui.get_color_u32_rgba(*cfg.COLOUR_NEUTRAL))
        drawlist.add_rect_filled(8 + origin[0], y,
                                 8 + origin[0] + max(0, cw * min([1.0, process.progress]) - cancel_button_w),
                                 y + height,
                                 imgui.get_color_u32_rgba(*colour))
        imgui.dummy(cw - cancel_button_w, height)
        retval = False
        if cancellable:
            imgui.same_line(spacing=5)
            retval = imgui.image_button(self.icon_close.renderer_id, SegmentationEditor.PROGRESS_BAR_HEIGHT, SegmentationEditor.PROGRESS_BAR_HEIGHT)
        return retval

    def _gui_feature_title_context_menu(self, feature_or_model):
        if imgui.begin_popup_context_item():
            for t in self.feature_colour_dict:
                imgui.selectable(t)
                if imgui.is_item_hovered():
                    feature_or_model.title = t
                    feature_or_model.colour = self.feature_colour_dict[t]
                imgui.spacing()

            imgui.end_popup()

    @staticmethod
    def tooltip(text):
        if imgui.is_item_hovered() and not imgui.is_mouse_down(0):
            if SegmentationEditor.TOOLTIP_HOVERED_TIMER == 0.0:
                SegmentationEditor.TOOLTIP_HOVERED_START_TIME = time.time()
                SegmentationEditor.TOOLTIP_HOVERED_TIMER = 0.001  # add a fake 1 ms to get bypass of this if clause in the next iter.
            elif SegmentationEditor.TOOLTIP_HOVERED_TIMER > SegmentationEditor.TOOLTIP_APPEAR_DELAY:
                imgui.set_tooltip(text)
            else:
                SegmentationEditor.TOOLTIP_HOVERED_TIMER = time.time() - SegmentationEditor.TOOLTIP_HOVERED_START_TIME
        if not imgui.is_any_item_hovered():
            SegmentationEditor.TOOLTIP_HOVERED_TIMER = 0.0

    def launch_export_coordinates(self, feature_title):
        try:
            if not os.path.isdir(self.export_dir):
                os.makedirs(self.export_dir)
            datasets = [d for d in cfg.se_frames if d.export]
            if not datasets:
                return

            feature_tag = f"_{feature_title}"
            for d in datasets:
                feature_mrc_path = os.path.splitext(d.path)[0]+feature_tag+".mrc"
                if os.path.isfile(feature_mrc_path):
                    self.queued_extracts.append(QueuedExtract(feature_mrc_path, SegmentationEditor.EXTRACT_THRESHOLD, SegmentationEditor.EXTRACT_MIN_WEIGHT, SegmentationEditor.EXTRACT_MIN_SPACING, self.export_dir))
                else:
                    print(f"Feature with tag {feature_title} not found for dataset at {d.path}")
            if self.queued_extracts:
                self.queued_extracts[0].start()
        except Exception as e:
            cfg.set_error(e, "Could not launch QueuedExtract jobs - see details below:")

    def launch_export_volumes(self):
        try:
            if not os.path.isdir(self.export_dir):
                os.makedirs(self.export_dir)
            datasets = [d for d in cfg.se_frames if d.export]
            models = [m for m in cfg.se_models if m.export]
            if not datasets:
                return
            for d in datasets:
                self.queued_exports.append(QueuedExport(self.export_dir, d, models, self.export_batch_size, self.export_overlays))

            if self.queued_exports:
                self.queued_exports[0].start()
        except Exception as e:
            cfg.set_error(e, "Could not export volumes - see details below:")

    def launch_create_training_set(self):
        positive_feature_names = list()
        negative_feature_names = list()
        for f in self.all_feature_names:
            if self.trainset_feature_selection[f] == 1:
                positive_feature_names.append(f)
            elif self.trainset_feature_selection[f] == -1:
                negative_feature_names.append(f)
        if len(positive_feature_names) == 0:
            return
        path = filedialog.asksaveasfilename(filetypes=[("scNodes traindata", cfg.filetype_traindata)], initialfile=f"{self.trainset_boxsize}_{self.trainset_apix:.3f}_{positive_feature_names[0]}")
        if path == "":
            return
        if path[-len(cfg.filetype_traindata):] != cfg.filetype_traindata:
            path += cfg.filetype_traindata
        #


        datasets_to_sample = list()
        for dataset in cfg.se_frames:
            if dataset.sample:
                datasets_to_sample.append(dataset)

        n_boxes = self.trainset_num_boxes_positive + self.trainset_num_boxes_negative
        if n_boxes == 0:
            return
        args = (path, n_boxes, positive_feature_names, negative_feature_names, datasets_to_sample, self.trainset_boxsize, self.trainset_apix)

        process = BackgroundProcess(self._create_training_set, args)
        process.start()
        self.active_trainset_exports.append(process)

    def _create_training_set(self, path, n_boxes, positives, negatives, datasets, boxsize, apix, process):
        positive = list()
        negative = list()

        n_done = 0
        target_type_dict = {np.float32: float, float: float, np.dtype('int8'): np.dtype('float32'), np.dtype('int16'): np.dtype('float32')}
        for d in datasets:
            mrcf = mrcfile.mmap(d.path, mode="r", permissive=True)
            raw_type = mrcf.data.dtype
            out_type = float
            if raw_type in target_type_dict:
                out_type = target_type_dict[raw_type]

            w = d.width
            h = d.height
            crop_px = int(np.ceil((boxsize * apix) / (d.pixel_size * 10.0)))  # size to crop so that the crop contains at least a boxsize*apix sized region
            scale_fac = (d.pixel_size * 10.0) / apix  # how much to scale the cropped images.
            nm = int(np.floor(crop_px / 2))
            pm = int(np.ceil(crop_px / 2))
            is_positive = True
            # find all boxes
            for f in d.features:
                if f.title in positives:
                    is_positive = True
                elif f.title in negatives:
                    is_positive = False
                else:
                    continue

                # find boxes
                for z in f.boxes.keys():
                    if f.boxes[z] is not None:
                        for (x, y) in f.boxes[z]:
                            x_min = (x-nm)
                            x_max = (x+pm)
                            y_min = (y-nm)
                            y_max = (y+pm)
                            if x_min > 0 and y_min > 0 and x_max < w and y_max < h:
                                image = np.flipud(mrcf.data[z, y_min:y_max, x_min:x_max])
                                image = np.array(image.astype(out_type, copy=False), dtype=float)
                                image = zoom(image, scale_fac)
                                image = image[:boxsize, :boxsize]
                                if z in f.slices and f.slices[z] is not None and is_positive:
                                    segmentation = np.flipud(f.slices[z][y_min:y_max, x_min:x_max])
                                    segmentation = zoom(segmentation, scale_fac)
                                    segmentation = segmentation[:boxsize, :boxsize]
                                    segmentation = zoom(segmentation, scale_fac)
                                    segmentation = segmentation[:boxsize, :boxsize]
                                else:
                                    segmentation = np.zeros_like(image)
                                if is_positive:
                                    positive.append([image, segmentation])
                                else:
                                    negative.append([image, segmentation])
                            n_done += 1
                            process.set_progress(n_done / n_boxes)

        if not negative:
            all_imgs = np.array(positive)
        else:
            all_imgs = np.array(positive + negative)
        tifffile.imwrite(path, all_imgs, description=f"apix={apix}")

    @staticmethod
    def seframe_from_clemframe(clemframe):
        if not os.path.exists(clemframe.path):
            cfg.set_error(Exception(f"No file exists at path {clemframe.path}"), "Could not convert CLEMFrame to SEFrame - see below.")
            return
        new_se_frame = SEFrame(clemframe.path)
        new_se_frame.clem_frame = clemframe
        new_se_frame.pixel_size = clemframe.pixel_size
        new_se_frame.autocontrast = False
        new_se_frame.title = clemframe.title
        new_se_frame.contrast_lims = clemframe.contrast_lims
        new_se_frame.set_slice(clemframe.current_slice)

        cfg.se_frames.append(new_se_frame)
        SegmentationEditor.set_active_dataset(cfg.se_frames[-1])

    @staticmethod
    def load_model_group(path):
        try:
            root = os.path.dirname(path)
            with open(path, 'r') as infile:
                model_paths = json.load(infile)
                json_path = model_paths.pop(0)

                # load models
                for path in model_paths:
                    cfg.se_models.append(SEModel())
                    cfg.se_models[-1].load(root + "/" + path)

                # load interactions
                with open(root + "/" + json_path, 'r') as f:
                    interaction_dict_list = json.load(f)
                    for d in interaction_dict_list:
                        ModelInteraction.from_dict(d)

        except Exception as e:
            cfg.set_error(e, "Error loading model group, see details below.")

    @staticmethod
    def save_model_group(path):
        try:
            root = os.path.dirname(path)
            group_name = os.path.splitext(os.path.basename(path))[0]
            # save all models as '../[group_name]_model_N'
            model_paths = list()
            i = 0
            for m in cfg.se_models:
                m_path = group_name + f"_model_{i}" + cfg.filetype_semodel
                m.save(root + "/" + m_path)
                model_paths.append(m_path)
                i += 1

            # save interactions as json.
            interaction_dict_list = list()
            for interaction in ModelInteraction.all:
                interaction_dict_list.append(interaction.as_dict())
            json_path = f"{group_name}_interactions.json"
            with open(root + "/" + json_path, 'w') as outfile:
                json.dump(interaction_dict_list, outfile, indent=2)

            all_paths = [json_path] + model_paths
            with open(path, 'w') as outfile:
                json.dump(all_paths, outfile, indent=2)
        except Exception as e:
            cfg.set_error(e, "Error saving model group, see details below")

    def camera_control(self):
        if imgui.get_io().want_capture_mouse or imgui.get_io().want_capture_keyboard:
            return None
        if self.active_tab != "Pick":
            if self.window.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE):
                delta_cursor = self.window.cursor_delta
                self.camera.position[0] += delta_cursor[0] / self.camera.zoom
                self.camera.position[1] -= delta_cursor[1] / self.camera.zoom
            if self.window.get_key(glfw.KEY_LEFT_SHIFT):
                self.camera.zoom *= (1.0 + self.window.scroll_delta[1] * SegmentationEditor.CAMERA_ZOOM_STEP)
                self.camera.zoom = min([self.camera.zoom, SegmentationEditor.CAMERA_MAX_ZOOM])
        else:
            if self.window.get_key(glfw.KEY_LEFT_SHIFT):
                if self.window.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE):
                    self.camera3d.pitch += self.window.cursor_delta[1] * SegmentationEditor.VIEW_3D_PIVOT_SPEED
                    self.camera3d.yaw += self.window.cursor_delta[0] * SegmentationEditor.VIEW_3D_PIVOT_SPEED
                    self.camera3d.pitch = clamp(self.camera3d.pitch, -89.9, 89.9)
                    if self.window.cursor_delta[0] != 0 or self.window.cursor_delta[1] != 0:
                        SegmentationEditor.VIEW_REQUIRES_UPDATE = True
                self.camera3d.distance -= self.window.scroll_delta[1] * SegmentationEditor.VIEW_3D_MOVE_SPEED
                if self.window.scroll_delta[1] != 0:
                    SegmentationEditor.VIEW_REQUIRES_UPDATE = True
                self.camera3d.distance = max(10.0, self.camera3d.distance)
            elif self.window.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE):
                dx = -self.window.cursor_delta[0]
                dy = -self.window.cursor_delta[1]
                world_delta = self.camera3d.cursor_delta_to_world_delta((dx, dy))
                self.camera3d.focus[0] += world_delta[0]
                self.camera3d.focus[1] += world_delta[1]
                self.camera3d.focus[2] += world_delta[2]
                if dx != 0 or dy != 0:
                    SegmentationEditor.VIEW_REQUIRES_UPDATE = True

    def end_frame(self):
        self.window.end_frame()


class Brush:
    circular_roi = np.zeros(1, dtype=np.uint8)
    circular_roi_radius = -1

    @staticmethod
    def set_circular_roi_radius(radius):
        if Brush.circular_roi_radius == radius:
            return
        Brush.circular_roi_radius = radius
        Brush.circular_roi = np.zeros((2*radius+1, 2*radius+1), dtype=np.uint8)
        r = radius**2
        for x in range(0, 2*radius+1):
            for y in range(0, 2*radius+1):
                if ((x-radius)**2 + (y-radius)**2) < r:
                    Brush.circular_roi[x, y] = True

    @staticmethod
    def apply_circular(segmentation, center_coordinates, val=True):
        # check if the current slice already exists; if not, make it.
        segmentation.request_draw_in_current_slice()
        r = int(segmentation.brush_size)
        center_coordinates[0], center_coordinates[1] = center_coordinates[1], center_coordinates[0]
        Brush.set_circular_roi_radius(r)

        x = [center_coordinates[0] - r, center_coordinates[0] + r + 1]
        y = [center_coordinates[1] - r, center_coordinates[1] + r + 1]
        rx = [0, 2 * r + 1]
        ry = [0, 2 * r + 1]
        if x[0] > segmentation.height or x[1] < 0 or y[0] > segmentation.width or y[1] < 0:
            return
        if x[0] < 0:
            rx[0] -= x[0]
            x[0] = 0
        if y[0] < 0:
            ry[0] -= y[0]
            y[0] = 0
        if x[1] > segmentation.height:
            rx[1] -= (x[1] - segmentation.height)
            x[1] = segmentation.height
        if y[1] > segmentation.width:
            ry[1] -= (y[1] - segmentation.width)
            y[1] = segmentation.width

        if val:
            segmentation.data[x[0]:x[1], y[0]:y[1]] += Brush.circular_roi[rx[0]:rx[1], ry[0]:ry[1]]
        else:
            segmentation.data[x[0]:x[1], y[0]:y[1]] *= (np.uint8(1.0) - Brush.circular_roi[rx[0]:rx[1], ry[0]:ry[1]])
        segmentation.data[x[0]:x[1], y[0]:y[1]] = np.clip(segmentation.data[x[0]:x[1], y[0]:y[1]], 0, 1)
        segmentation.texture.update_subimage(segmentation.data[x[0]:x[1], y[0]:y[1]], y[0], x[0])


class Renderer:
    def __init__(self):
        self.quad_shader = Shader(os.path.join(cfg.root, "shaders", "se_quad_shader.glsl"))
        self.b_segmentation_shader = Shader(os.path.join(cfg.root, "shaders", "se_binary_segmentation_shader.glsl"))
        self.f_segmentation_shader = Shader(os.path.join(cfg.root, "shaders", "se_float_segmentation_shader.glsl"))
        self.overlay_shader = Shader(os.path.join(cfg.root, "shaders", "se_overlay_shader.glsl"))
        self.border_shader = Shader(os.path.join(cfg.root, "shaders", "se_border_shader.glsl"))
        self.kernel_filter = Shader(os.path.join(cfg.root, "shaders", "se_compute_kernel_filter.glsl"))
        self.mix_filtered = Shader(os.path.join(cfg.root, "shaders", "se_compute_mix.glsl"))
        self.line_shader = Shader(os.path.join(cfg.root, "shaders", "ce_line_shader.glsl"))
        self.icon_shader = Shader(os.path.join(cfg.root, "shaders", "se_icon_shader.glsl"))
        self.surface_model_shader = Shader(os.path.join(cfg.root, "shaders", "se_surface_model_shader.glsl"))
        self.line_3d_shader = Shader(os.path.join(cfg.root, "shaders", "se_line_3d_shader.glsl"))
        self.quad_3d_shader = Shader(os.path.join(cfg.root, "shaders", "se_quad_3d_shader.glsl"))
        self.depth_mask_shader = Shader(os.path.join(cfg.root, "shaders", "se_depth_mask_shader.glsl"))
        self.ray_trace_shader = Shader(os.path.join(cfg.root, "shaders", "se_overlay_ray_trace_shader.glsl"))
        self.overlay_blend_shader = Shader(os.path.join(cfg.root, "shaders", "se_overlay_blend_shader.glsl"))
        self.line_list = list()
        self.line_list_s = list()
        self.line_va = VertexArray(None, None, attribute_format="xyrgb")
        self.fbo1 = FrameBuffer()
        self.fbo2 = FrameBuffer()
        self.fbo3 = FrameBuffer(100, 100)
        self.ray_trace_fbo_a = FrameBuffer()
        self.ray_trace_fbo_b = FrameBuffer()
        self.ndc_screen_va = VertexArray(attribute_format="xy")
        self.ndc_screen_va.update(VertexBuffer([-1, -1, 1, -1, 1, 1, -1, 1]), IndexBuffer([0, 1, 2, 0, 2, 3]))
        self.ray_trace_fbo_size = [0.0, 0.0]

    def recompile_shaders(self):  # for debugging
        try:
            self.quad_shader = Shader(os.path.join(cfg.root, "shaders", "se_quad_shader.glsl"))
            self.b_segmentation_shader = Shader(os.path.join(cfg.root, "shaders", "se_binary_segmentation_shader.glsl"))
            self.f_segmentation_shader = Shader(os.path.join(cfg.root, "shaders", "se_float_segmentation_shader.glsl"))
            self.overlay_shader = Shader(os.path.join(cfg.root, "shaders", "se_overlay_shader.glsl"))
            self.border_shader = Shader(os.path.join(cfg.root, "shaders", "se_border_shader.glsl"))
            self.kernel_filter = Shader(os.path.join(cfg.root, "shaders", "se_compute_kernel_filter.glsl"))
            self.mix_filtered = Shader(os.path.join(cfg.root, "shaders", "se_compute_mix.glsl"))
            self.line_shader = Shader(os.path.join(cfg.root, "shaders", "ce_line_shader.glsl"))
            self.icon_shader = Shader(os.path.join(cfg.root, "shaders", "se_icon_shader.glsl"))
            self.surface_model_shader = Shader(os.path.join(cfg.root, "shaders", "se_surface_model_shader.glsl"))
            self.line_3d_shader = Shader(os.path.join(cfg.root, "shaders", "se_line_3d_shader.glsl"))
            self.quad_3d_shader = Shader(os.path.join(cfg.root, "shaders", "se_quad_3d_shader.glsl"))
            self.depth_mask_shader = Shader(os.path.join(cfg.root, "shaders", "se_depth_mask_shader.glsl"))
            self.ray_trace_shader = Shader(os.path.join(cfg.root, "shaders", "se_overlay_ray_trace_shader.glsl"))
            self.overlay_blend_shader = Shader(os.path.join(cfg.root, "shaders", "se_overlay_blend_shader.glsl"))
        finally:
            pass

    def render_filtered_frame(self, se_frame, camera, window, filters, camera3d=None, emphasize_roi=False):
        se_frame.update_model_matrix()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)

        self.fbo1.clear((0.0, 0.0, 0.0, 1.0))
        self.fbo2.clear((0.0, 0.0, 0.0, 1.0))
        self.fbo3.clear((0.0, 0.0, 0.0, 2.0))

        # render the image to a framebuffer
        fake_camera_matrix = np.matrix(
            [[2 / self.fbo1.width, 0, 0, 0], [0, 2 / self.fbo1.height, 0, 0], [0, 0, -2 / 100, 0], [0, 0, 0, 1]])
        self.fbo1.bind()
        self.quad_shader.bind()
        se_frame.quad_va.bind()
        se_frame.texture.bind(0)
        self.quad_shader.uniformmat4("cameraMatrix", fake_camera_matrix)
        self.quad_shader.uniformmat4("modelMatrix", np.identity(4))
        self.quad_shader.uniform1f("alpha", se_frame.alpha)
        self.quad_shader.uniform1f("contrastMin", 0.0)
        self.quad_shader.uniform1f("contrastMax", 1.0)
        glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.quad_shader.unbind()
        se_frame.quad_va.unbind()
        glActiveTexture(GL_TEXTURE0)
        self.fbo1.unbind()
        window.set_full_viewport()
        # filter framebuffer
        self.kernel_filter.bind()
        compute_size = (int(np.ceil(se_frame.width / 16)), int(np.ceil(se_frame.height / 16)), 1)
        for fltr in filters:
            if not fltr.enabled:
                continue
            self.kernel_filter.bind()

            # horizontal shader pass
            fltr.bind(horizontal=True)
            glBindImageTexture(0, self.fbo1.texture.renderer_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
            glBindImageTexture(1, self.fbo2.texture.renderer_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)
            self.kernel_filter.uniform1i("direction", 0)
            glDispatchCompute(*compute_size)
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

            # vertical shader pass
            fltr.bind(horizontal=False)
            glBindImageTexture(0, self.fbo2.texture.renderer_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
            glBindImageTexture(1, self.fbo3.texture.renderer_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)
            self.kernel_filter.uniform1i("direction", 1)
            glDispatchCompute(*compute_size)
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
            fltr.unbind()
            self.kernel_filter.unbind()

            ## mix the filtered and the original image
            self.mix_filtered.bind()
            glBindImageTexture(0, self.fbo3.texture.renderer_id, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)
            glBindImageTexture(1, self.fbo1.texture.renderer_id, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)
            self.mix_filtered.uniform1f("strength", fltr.strength)
            glDispatchCompute(*compute_size)
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
            self.mix_filtered.unbind()

        pxd = None
        if se_frame.requires_histogram_update:
            self.fbo1.bind()
            pxd = glReadPixels(0, 0, self.fbo1.width, self.fbo1.height, GL_RED, GL_FLOAT)
            self.fbo1.unbind()
            window.set_full_viewport()

        # render the framebuffer to the screen

        shader = self.quad_shader if not camera3d else self.quad_3d_shader
        vpmat = camera.view_projection_matrix if not camera3d else camera3d.matrix
        shader.bind()
        se_frame.quad_va.bind()
        self.fbo1.texture.bind(0)
        shader.uniformmat4("cameraMatrix", vpmat)
        shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        crop_x_lims = [se_frame.crop_roi[0] / se_frame.width, 1.0 - (se_frame.width - se_frame.crop_roi[2]) / se_frame.width]
        crop_y_lims = [(se_frame.height - se_frame.crop_roi[3]) / se_frame.height, 1.0 - se_frame.crop_roi[1] / se_frame.height]
        if not emphasize_roi:
            crop_x_lims = [0, 1]
            crop_y_lims = [0, 1]
        shader.uniform2f("xLims", crop_x_lims)
        shader.uniform2f("yLims", crop_y_lims)
        shader.uniform1f("alpha", se_frame.alpha if camera3d is None else SegmentationEditor.PICKING_FRAME_ALPHA)
        if camera3d is not None:
            glEnable(GL_DEPTH_TEST)
            shader.uniform1f("z_pos", (se_frame.current_slice - se_frame.n_slices / 2) * se_frame.pixel_size)
            shader.uniform1f("pixel_size", se_frame.pixel_size)  # This factor should be in the VA, but fixing that messed up the 2D view - cheap fix for now
        if se_frame.invert:
            shader.uniform1f("contrastMin", se_frame.contrast_lims[1])
            shader.uniform1f("contrastMax", se_frame.contrast_lims[0])
        else:
            shader.uniform1f("contrastMin", se_frame.contrast_lims[0])
            shader.uniform1f("contrastMax", se_frame.contrast_lims[1])
        glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        shader.unbind()
        se_frame.quad_va.unbind()
        glActiveTexture(GL_TEXTURE0)
        if camera3d is not None:
            glDisable(GL_DEPTH_TEST)
        return pxd

    def render_models(self, se_frame, camera):
        # render overlays (from models)
        se_frame.quad_va.bind()
        self.f_segmentation_shader.bind()
        self.f_segmentation_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.f_segmentation_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        render_list = copy(cfg.se_models)
        if cfg.se_active_model in render_list:
            render_list.remove(cfg.se_active_model)
            render_list.append(cfg.se_active_model)
        for model in render_list:
            if not model.active or not model.show:
                continue
            if model.data is None:
                continue
            if model.blend:
                glBlendFunc(GL_DST_COLOR, GL_DST_ALPHA)
                glBlendEquation(GL_FUNC_ADD)
            else:
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glBlendEquation(GL_FUNC_ADD)
            self.f_segmentation_shader.uniform1f("alpha", model.alpha)
            clr = model.colour
            alpha = model.alpha
            self.f_segmentation_shader.uniform3f("colour", (clr[0] * alpha, clr[1] * alpha, clr[2] * alpha))
            self.f_segmentation_shader.uniform1f("threshold", model.threshold)
            model.texture.bind(0)
            glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        se_frame.quad_va.unbind()
        self.f_segmentation_shader.unbind()
        glActiveTexture(GL_TEXTURE0)

        # render border
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)
        self.border_shader.bind()
        se_frame.border_va.bind()
        self.border_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.border_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        self.border_shader.uniform1f("z_pos", 0)
        self.border_shader.uniform1f("alpha", 1.0)
        glDrawElements(GL_LINES, se_frame.border_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.border_shader.unbind()
        se_frame.border_va.unbind()

    def render_segmentations(self, se_frame, camera):
        """
        se_frame: the SEFrame object to render.
        camera: a Camera object to render with.
        window: the Window object (to which the viewport size will be reset)
        filters: a list of Filter objectm, to apply to the se_frame pixeldata.
        """

        # render segmentation overlays
        se_frame.quad_va.bind()
        self.b_segmentation_shader.bind()
        self.b_segmentation_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.b_segmentation_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        render_list = copy(se_frame.features)
        if se_frame.active_feature in render_list:
            render_list.remove(se_frame.active_feature)
            render_list.append(se_frame.active_feature)
        for segmentation in render_list:
            if segmentation.hide:
                continue
            if segmentation.data is None:
                continue
            if not segmentation.contour:
                glBlendFunc(GL_DST_COLOR, GL_DST_ALPHA)
                glBlendEquation(GL_FUNC_ADD)
            else:
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glBlendEquation(GL_FUNC_ADD)
            self.b_segmentation_shader.uniform1f("alpha", segmentation.alpha)
            clr = segmentation.colour
            alpha = segmentation.alpha
            self.b_segmentation_shader.uniform3f("colour", (clr[0] * alpha, clr[1] * alpha, clr[2] * alpha))
            self.b_segmentation_shader.uniform1i("contour", int(segmentation.contour))

            segmentation.texture.bind(0)
            glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        se_frame.quad_va.unbind()
        self.b_segmentation_shader.unbind()
        glActiveTexture(GL_TEXTURE0)

        # render border
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)
        self.border_shader.bind()
        se_frame.border_va.bind()
        self.border_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.border_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        self.border_shader.uniform1f("z_pos", 0)
        self.border_shader.uniform1f("alpha", 1.0)
        glDrawElements(GL_LINES, se_frame.border_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.border_shader.unbind()
        se_frame.border_va.unbind()

    def render_frame_border(self, se_frame, camera):
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)
        self.border_shader.bind()
        se_frame.border_va.bind()
        self.border_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.border_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        self.border_shader.uniform1f("z_pos", (se_frame.current_slice - se_frame.n_slices / 2))
        self.border_shader.uniform1f("alpha", SegmentationEditor.PICKING_FRAME_ALPHA)
        glDrawElements(GL_LINES, se_frame.border_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.border_shader.unbind()
        se_frame.border_va.unbind()

    def render_crop_handles(self, se_frame, camera, handles):
        self.icon_shader.bind()
        h_va = handles[0].get_va()
        h_va.bind()
        handles[0].get_texture().bind(0)
        self.icon_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        for h in handles:
            h.compute_matrix(se_frame, camera)
            self.icon_shader.uniformmat4("modelMatrix", h.transform.matrix)
            self.icon_shader.uniform1i("invert", int(se_frame.invert))
            glDrawElements(GL_TRIANGLES, h_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.icon_shader.unbind()
        h_va.unbind()

    def render_surface_models(self, surface_models, camera, ambient_strength, spot_light):
        glEnable(GL_DEPTH_TEST)
        self.surface_model_shader.bind()
        self.surface_model_shader.uniformmat4("vpMat", camera.matrix)
        self.surface_model_shader.uniform3f("viewDir", camera.get_view_direction())
        self.surface_model_shader.uniform3f("lightDir", spot_light.vec)
        self.surface_model_shader.uniform1f("ambientStrength", ambient_strength)
        self.surface_model_shader.uniform1f("lightStrength", spot_light.strength)
        self.surface_model_shader.uniform3f("lightColour", spot_light.colour)
        self.surface_model_shader.uniform1i("style", SegmentationEditor.SELECTED_RENDER_STYLE)
        glEnable(GL_DEPTH_TEST)
        for s in surface_models:
            if s.hide:
                continue
            self.surface_model_shader.uniform4f("color", [*s.colour, s.alpha])
            for blob in s.blobs.values():
                if blob.complete and not blob.hide:
                    blob.va.bind()
                    glDrawElements(GL_TRIANGLES, blob.va.indexBuffer.getCount(), GL_UNSIGNED_INT, None)
                    blob.va.unbind()
        self.surface_model_shader.unbind()
        glDisable(GL_DEPTH_TEST)

    def render_line_va(self, va, camera):
        glEnable(GL_DEPTH_TEST)
        self.line_3d_shader.bind()
        self.line_3d_shader.uniformmat4("cameraMatrix", camera.matrix)
        va.bind()
        glDrawElements(GL_LINES, va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        va.unbind()
        self.line_3d_shader.unbind()
        glDisable(GL_DEPTH_TEST)

    def ray_trace_overlay(self, window_size, se_frame, camera, box_va):
        if se_frame.overlay is None:
            print("SEFrame doesn't have an overlay; remove function call, wherever it was.")
            return

        # 1 - Rendering a depth mask to find where to START sampling rays.
        if self.ray_trace_fbo_size != window_size:
            self.ray_trace_fbo_size = window_size
            self.ray_trace_fbo_a = FrameBuffer(window_size[0], window_size[1], "rgba32f")
            self.ray_trace_fbo_b = FrameBuffer(window_size[0], window_size[1], "rgba32f")
        self.ray_trace_fbo_a.bind()  # UNBIND FBO in such a way that the original rendered scene is the new fbo again.
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
        glClear(GL_DEPTH_BUFFER_BIT)

        self.depth_mask_shader.bind()
        self.depth_mask_shader.uniformmat4("vpMat", camera.matrix)
        box_va.bind()
        glDrawElements(GL_TRIANGLES, box_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.depth_mask_shader.unbind()
        box_va.unbind()
        self.ray_trace_fbo_a.unbind((0, 0, window_size[0], window_size[1]))

        # - Rendering a depth mask to find where to STOP sampling rays
        self.ray_trace_fbo_b.bind()
        glClearDepth(0.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_GREATER)
        glDepthMask(GL_TRUE)
        glClear(GL_DEPTH_BUFFER_BIT)

        self.depth_mask_shader.bind()
        self.depth_mask_shader.uniformmat4("vpMat", camera.matrix)
        box_va.bind()
        glDrawElements(GL_TRIANGLES, box_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.depth_mask_shader.unbind()
        box_va.unbind()
        self.ray_trace_fbo_b.unbind((0, 0, window_size[0], window_size[1]))
        glDepthFunc(GL_LESS)
        glClearDepth(1.0)
        # 2 - ray tracing
        self.ray_trace_shader.bind()
        self.ray_trace_shader.uniformmat4("ipMat", camera.ipmat)
        self.ray_trace_shader.uniformmat4("ivMat", camera.ivmat)
        self.ray_trace_shader.uniformmat4("pMat", camera.pmat)
        self.ray_trace_shader.uniform1f("near", camera.clip_near)
        self.ray_trace_shader.uniform1f("far", camera.clip_far)
        self.ray_trace_shader.uniform2f("viewportSize", window_size)
        self.ray_trace_shader.uniform1f("zLim", se_frame.n_slices * se_frame.pixel_size / 2.0)
        self.ray_trace_shader.uniform1f("zQuad", (se_frame.current_slice - se_frame.n_slices /2) * se_frame.pixel_size)
        self.ray_trace_shader.uniform1f("pixelSize", se_frame.pixel_size)
        self.ray_trace_shader.uniform2f("imgSize", [se_frame.overlay.size[1], se_frame.overlay.size[0]])
        self.ray_trace_shader.uniform1i("style", SegmentationEditor.BLEND_MODES_3D[SegmentationEditor.BLEND_MODES_LIST_3D[SegmentationEditor.OVERLAY_BLEND_MODE_3D]][3])
        self.ray_trace_shader.uniform1f("intensity", SegmentationEditor.OVERLAY_INTENSITY)
        se_frame.overlay.texture.bind(0)  # overlay image, to read from
        glActiveTexture(GL_TEXTURE0 + 1)  # depth buffer, to read from
        glBindTexture(GL_TEXTURE_2D, self.ray_trace_fbo_a.depth_texture_renderer_id)
        glActiveTexture(GL_TEXTURE0 + 2)
        glBindTexture(GL_TEXTURE_2D, self.ray_trace_fbo_b.depth_texture_renderer_id)
        self.ray_trace_fbo_a.texture.bind_image_slot(3, 1)  # fbo texture, to write to
        glDispatchCompute(int(window_size[0] // 16) + 1, int(window_size[1] // 16) + 1, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        SegmentationEditor.VIEW_REQUIRES_UPDATE = False

    def render_overlay_3d(self, alpha, intensity):
        # add overlay image to whatever has been rendered before.
        blend_mode = SegmentationEditor.BLEND_MODES[SegmentationEditor.BLEND_MODES_LIST[SegmentationEditor.OVERLAY_BLEND_MODE]]
        glBlendFunc(blend_mode[0], blend_mode[1])
        glBlendEquation(blend_mode[2])
        glDisable(GL_DEPTH_TEST)
        self.overlay_blend_shader.bind()
        self.overlay_blend_shader.uniform1f("alpha", alpha)
        self.overlay_blend_shader.uniform1f("intensity", intensity)
        self.ray_trace_fbo_a.texture.bind(0)
        self.ndc_screen_va.bind()
        glDrawElements(GL_TRIANGLES, self.ndc_screen_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.ndc_screen_va.unbind()
        self.overlay_blend_shader.unbind()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)

    def add_line(self, start_xy, stop_xy, colour, subtract=False):
        if subtract:
            self.line_list_s.append((start_xy, stop_xy, colour))
        else:
            self.line_list.append((start_xy, stop_xy, colour))

    def add_circle(self, center_xy, radius, colour, segments=32, subtract=False):
        theta = 0
        start_xy = (center_xy[0] + radius * np.cos(theta), center_xy[1] + radius * np.sin(theta))
        for i in range(segments):
            theta = (i * 2 * np.pi / (segments - 1))
            stop_xy = (center_xy[0] + radius * np.cos(theta), center_xy[1] + radius * np.sin(theta))
            if subtract:
                self.line_list_s.append((start_xy, stop_xy, colour))
            else:
                self.line_list.append((start_xy, stop_xy, colour))
            start_xy = (center_xy[0] + radius * np.cos(theta), center_xy[1] + radius * np.sin(theta))

    def add_square(self, center_xy, size, colour, subtract=False):
        bottom = center_xy[1] - size / 2
        top = center_xy[1] + size / 2
        left = center_xy[0] - size / 2
        right = center_xy[0] + size / 2
        if subtract:
            self.line_list_s.append(((left, bottom), (right, bottom), colour))
            self.line_list_s.append(((left, top), (right, top), colour))
            self.line_list_s.append(((left, bottom), (left, top), colour))
            self.line_list_s.append(((right, bottom), (right, top), colour))
        else:
            self.line_list.append(((left, bottom), (right, bottom), colour))
            self.line_list.append(((left, top), (right, top), colour))
            self.line_list.append(((left, bottom), (left, top), colour))
            self.line_list.append(((right, bottom), (right, top), colour))

    def render_lines(self, camera):
        def render_lines_in_list(line_list):
            # make VA
            vertices = list()
            indices = list()
            i = 0
            for line in line_list:
                vertices += [*line[0], *line[2][:3]]
                vertices += [*line[1], *line[2][:3]]
                indices += [2*i, 2*i+1]
                i += 1
            self.line_va.update(VertexBuffer(vertices), IndexBuffer(indices))

            # launch
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glBlendEquation(GL_FUNC_ADD)
            self.line_shader.bind()
            self.line_va.bind()
            self.line_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
            glDrawElements(GL_LINES, self.line_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            self.line_shader.unbind()
            self.line_va.unbind()

        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR)
        glBlendEquation(GL_FUNC_ADD)
        render_lines_in_list(self.line_list_s)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)
        render_lines_in_list(self.line_list)

        self.line_list_s = list()
        self.line_list = list()

    def render_draw_list(self, camera):
        self.render_lines(camera)

    def render_overlay(self, se_frame, camera, blend_mode, alpha):
        if se_frame.overlay is None:
            return
        if alpha == 0.0:
            return
        glBlendFunc(blend_mode[0], blend_mode[1])
        glBlendEquation(blend_mode[2])
        shader_blend_code = blend_mode[3]
        se_frame.quad_va.bind()
        se_frame.overlay.texture.bind()
        self.overlay_shader.bind()
        self.overlay_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.overlay_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        self.overlay_shader.uniform1f("alpha", alpha)
        self.overlay_shader.uniform1i("shader_blend_code", shader_blend_code)
        glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.overlay_shader.unbind()
        se_frame.quad_va.unbind()
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


class Light3D:
    def __init__(self):
        self.colour = (1.0, 1.0, 1.0)
        self.vec = (0.0, 1.0, 0.0)
        self.yaw = 20.0
        self.pitch = 0.0
        self.strength = 0.5

    def compute_vec(self, dyaw=0, dpitch=0):
        # Calculate the camera forward vector based on pitch and yaw
        cos_pitch = np.cos(np.radians(self.pitch + dpitch))
        sin_pitch = np.sin(np.radians(self.pitch + dpitch))
        cos_yaw = np.cos(np.radians(self.yaw + dyaw))
        sin_yaw = np.sin(np.radians(self.yaw + dyaw))

        forward = np.array([-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw])
        self.vec = forward


class Camera3D:
    def __init__(self):
        self.view_matrix = np.eye(4)
        self.projection_matrix = np.eye(4)
        self.view_projection_matrix = np.eye(4)
        self.focus = np.zeros(3)
        self.pitch = 0.0
        self.yaw = 180.0
        self.distance = 1120.0
        self.clip_near = 1e-1
        self.clip_far = 1e4
        self.projection_width = 1
        self.projection_height = 1
        self.set_projection_matrix(cfg.window_width, cfg.window_height)

    def set_projection_matrix(self, window_width, window_height):
        self.projection_width = window_width
        self.projection_height = window_height
        self.update_projection_matrix()

    def cursor_delta_to_world_delta(self, cursor_delta):
        self.yaw *= -1
        camera_right = np.cross([0, 1, 0], self.get_forward())
        camera_up = np.cross(camera_right, self.get_forward())
        self.yaw *= -1
        return cursor_delta[0] * camera_right + cursor_delta[1] * camera_up

    def get_forward(self):
        # Calculate the camera forward vector based on pitch and yaw
        cos_pitch = np.cos(np.radians(self.pitch))
        sin_pitch = np.sin(np.radians(self.pitch))
        cos_yaw = np.cos(np.radians(self.yaw))
        sin_yaw = np.sin(np.radians(self.yaw))

        forward = np.array([-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw])
        return forward

    @property
    def matrix(self):
        return self.view_projection_matrix

    @property
    def vpmat(self):
        return self.view_projection_matrix

    @property
    def ivpmat(self):
        return np.linalg.inv(self.view_projection_matrix)

    @property
    def pmat(self):
        return self.projection_matrix

    @property
    def vmat(self):
        return self.view_matrix

    @property
    def ipmat(self):
        return np.linalg.inv(self.projection_matrix)

    @property
    def ivmat(self):
        return np.linalg.inv(self.view_matrix)

    def on_update(self):
        self.update_projection_matrix()
        self.update_view_projection_matrix()

    def update_projection_matrix(self):
        aspect_ratio = self.projection_width / self.projection_height
        self.projection_matrix = Camera3D.create_perspective_matrix(60.0, aspect_ratio, self.clip_near, self.clip_far)
        self.update_view_projection_matrix()

    @staticmethod
    def create_perspective_matrix(fov, aspect_ratio, near, far):
        S = 1 / (np.tan(0.5 * fov / 180.0 * np.pi))
        f = far
        n = near

        projection_matrix = np.zeros((4, 4))
        projection_matrix[0, 0] = S / aspect_ratio
        projection_matrix[1, 1] = S
        projection_matrix[2, 2] = -f / (f - n)
        projection_matrix[3, 2] = -1
        projection_matrix[2, 3] = -f * n / (f - n)

        return projection_matrix

    def update_view_projection_matrix(self):
        eye_position = self.calculate_relative_position(self.focus, self.pitch, self.yaw, self.distance)
        self.view_matrix = self.create_look_at_matrix(eye_position, self.focus)
        self.view_projection_matrix = self.projection_matrix @ self.view_matrix

    def get_view_direction(self):
        eye_position = self.calculate_relative_position(self.focus, self.pitch, self.yaw, self.distance)
        focus_position = np.array(self.focus)
        view_dir = eye_position - focus_position
        view_dir /= np.sum(view_dir**2)**0.5
        return view_dir

    @staticmethod
    def calculate_relative_position(base_position, pitch, yaw, distance):
        cos_pitch = np.cos(np.radians(pitch))
        sin_pitch = np.sin(np.radians(pitch))
        cos_yaw = np.cos(np.radians(yaw))
        sin_yaw = np.sin(np.radians(yaw))

        forward = np.array([
            cos_pitch * sin_yaw,
            sin_pitch,
            -cos_pitch * cos_yaw
        ])
        forward = forward / np.linalg.norm(forward)

        relative_position = base_position + forward * distance

        return relative_position

    @staticmethod
    def create_look_at_matrix(eye, position):
        forward = Camera3D.normalize(position - eye)
        right = Camera3D.normalize(np.cross(forward, np.array([0, 1, 0])))
        up = np.cross(right, forward)

        look_at_matrix = np.eye(4)
        look_at_matrix[0, :3] = right
        look_at_matrix[1, :3] = up
        look_at_matrix[2, :3] = -forward
        look_at_matrix[:3, 3] = -np.dot(look_at_matrix[:3, :3], eye)
        return look_at_matrix

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm


class QueuedExport:
    def __init__(self, directory, dataset, models, batch_size, export_overlay):
        self.directory = directory
        self.dataset = dataset
        self.models = models
        self.export_overlay = export_overlay
        self.process = BackgroundProcess(self.do_export, (), name=f"{self.dataset.title} export")
        if self.models != []:
            self.colour = self.models[0].colour
        else:
            self.colour = (0.0, 0.0, 0.0)
        self.batch_size = batch_size

    def do_export(self, process):
        try:
            process.set_progress(0.0)
            start_time = time.time()
            print(f"QueuedExport - loading dataset {self.dataset.path}")
            rx, ry = self.dataset.get_roi_indices()
            mrcd = np.array(mrcfile.open(self.dataset.path, mode='r', permissive=True).data[:, :, :])
            target_type_dict = {np.float32: np.dtype('float32'), float: np.dtype('float32'), np.dtype('int8'): np.dtype('float32'), np.dtype('int16'): np.dtype('float32')}
            if mrcd.dtype not in target_type_dict:
                mrcd = mrcd.astype(float, copy=False)
            else:
                mrcd = np.array(mrcd.astype(target_type_dict[mrcd.dtype], copy=False), dtype=np.dtype('float32'))
            self.check_stop_request()
            n_slices = mrcd.shape[0]
            n_slices_total = (self.dataset.export_top - self.dataset.export_bottom) * max(1, len(self.models))
            n_slices_complete = 0
            segmentations = np.zeros((len(self.models), *mrcd.shape), dtype=np.uint8)
            m_idx = 0
            for m in self.models:
                print(f"QueuedExport - applying model {m.info}")
                self.colour = m.colour
                slices_to_process = list(range(self.dataset.export_bottom, self.dataset.export_top))
                while len(slices_to_process) > 0:
                    self.check_stop_request()
                    indices = list()
                    images = list()
                    for i in range(self.batch_size):
                        if len(slices_to_process) > 0:
                            indices.append(slices_to_process.pop(0))
                            images.append(mrcd[indices[-1], rx[0]:rx[1], ry[0]:ry[1]])
                    seg_images = m.apply_to_multiple_slices(images, self.dataset.pixel_size)
                    for i in range(len(indices)):
                        segmentations[m_idx, indices[i], rx[0]:rx[1], ry[0]:ry[1]] = (seg_images[i] * 255).astype(np.uint8)
                        n_slices_complete += 1
                        self.process.set_progress(min([0.999, n_slices_complete / n_slices_total]))

                m_idx += 1
            self.check_stop_request()
            # apply competition
            print(f"QueuedExport - model competition")
            emission_indices = list()
            absorption_indices = list()
            i = 0
            for m in self.models:
                if m.emit:
                    emission_indices.append(i)
                if m.absorb:
                    absorption_indices.append(i)
                i += 1
            if len(emission_indices) >= 1 and len(absorption_indices) >= 1:
                max_map = np.max(segmentations[emission_indices, :, :, :], axis=0)
                for i in absorption_indices:
                    self.check_stop_request()
                    segmentations[i, :, :, :][segmentations[i, :, :, :] < max_map] = 0

            # apply interactions
            print(f"QueuedExport - model interactions")
            for interaction in ModelInteraction.all:
                target_model = interaction.parent
                source_model = interaction.partner
                if target_model in self.models and source_model in self.models:
                    target_idx = self.models.index(target_model)
                    source_idx = self.models.index(source_model)
                    for i in range(self.dataset.export_bottom, self.dataset.export_top):
                        self.check_stop_request()
                        segmentations[target_idx, i, :, :] = interaction.apply_to_images(self.dataset.pixel_size, segmentations[source_idx, i, :, :], segmentations[target_idx, i, :, :])

            # save the mrc files
            i = 0
            self.colour = cfg.COLOUR_POSITIVE[0:3]
            for m in self.models:
                self.check_stop_request()
                print(f"QueuedExport - saving output of model {m.info}")
                out_path = os.path.join(self.directory, self.dataset.title+"_"+m.title+".mrc")
                with mrcfile.new(out_path, overwrite=True) as mrc:
                    s = segmentations[i, :, :, :].squeeze()
                    mrc.set_data(s)
                    mrc.voxel_size = self.dataset.pixel_size * 10.0
                n_slices_complete += n_slices
                i += 1

            # export overlay
            self.check_stop_request()
            if self.export_overlay and self.dataset.overlay is not None:
                print(f"QueuedExport - exporting overlay")
                overlay_grayscale = np.sum(self.dataset.overlay.pxd ** 2, axis=2)
                overlay_grayscale /= np.amax(overlay_grayscale)
                overlay_grayscale *= 255
                overlay_grayscale -= 128
                overlay_grayscale = overlay_grayscale.astype(np.int8)
                overlay_volume = np.tile(overlay_grayscale[np.newaxis, :, :], (n_slices, 1, 1))
                out_path = os.path.join(self.directory, self.dataset.title + "_overlay.mrc")
                with mrcfile.new(out_path, overwrite=True) as mrc:
                    mrc.set_data(overlay_volume)
                    mrc.voxel_size = self.dataset.pixel_size * 10.0

            self.process.set_progress(1.0)
            print(f"QueuedExport - done! ({time.time() - start_time:.2f} s.)\n")

        except Exception as e:
            cfg.set_error(e, "")
            print("An issue was encountered during export\n"
                  f"\tDataset: {self.dataset.path}\n"
                  f"\tError: ", e)
            self.process.set_progress(1.0)

    def check_stop_request(self):
        if self.process.stop_request.is_set():
            raise Exception("QueuedExport - process terminated by user. ")

    def start(self):
        self.process.start()

    def stop(self):
        if self.process.thread is not None:
            self.process.stop_request.set()


class QueuedExtract:
    def __init__(self, mrcpath, threshold, min_weight, min_spacing, save_dir):
        self.path = mrcpath
        self.threshold = threshold
        self.min_weight = min_weight
        self.min_spacing = min_spacing
        self.dir = save_dir
        self.process = BackgroundProcess(self.do_export, (), name=f"{self.path} find coords.")

    def do_export(self, process):
        try:
            get_maxima_3d_watershed(mrcpath=self.path, threshold=self.threshold, min_spacing=self.min_spacing, min_weight=self.min_weight, save_dir=self.dir, process=self.process)
        except Exception as e:
            cfg.set_error(e, "Error in QueuedExtract (finding coordinates in a .mrc) - see details below.")
        process.set_progress(1.0)

    def start(self):
        self.process.start()

    def stop(self):
        if self.process.thread is not None:
            self.process.stop_request.set()


class WorldSpaceIcon:
    # this class is only used to render and interact with the SegmentationEditor crop handles. fairly specific implementation
    crop_icon_va = None
    crop_icon_texture = None
    MIN_SIZE = 128

    def __init__(self, corner_idx=0):
        if WorldSpaceIcon.crop_icon_va is None:
            WorldSpaceIcon.init_opengl_objs()

        self.transform = Transform()
        self.corner_idx = corner_idx
        self.corner_positions_local = [[0, 0], [1, 0], [0, -1], [1, -1]]
        self.corner_positions = [[0, 0], [1, 0], [0, -1], [1, -1]]
        self.parent = None
        self.active = False
        self.moving_entire_roi = False

    def get_va(self):
        return WorldSpaceIcon.crop_icon_va

    def get_texture(self):
        return WorldSpaceIcon.crop_icon_texture

    @staticmethod
    def init_opengl_objs():
        # crop icon
        icon_vertices = [0, 0, 0, 0,
                         1, 0, 1, 0,
                         0, -1, 0, 1,
                         1, -1, 1, 1]
        icon_indices = [0, 1, 2, 2, 1, 3]
        WorldSpaceIcon.crop_icon_va = VertexArray(VertexBuffer(icon_vertices), IndexBuffer(icon_indices),
                                                  attribute_format="xyuv")
        WorldSpaceIcon.crop_icon_texture = Texture(format="rgba32f")
        pxd_icon_crop = np.asarray(Image.open(os.path.join(cfg.root, "icons", "icon_crop_256.png"))).astype(
            np.float32) / 255.0
        WorldSpaceIcon.crop_icon_texture.update(pxd_icon_crop)

    def affect_crop(self, pixel_coordinate):
        d = WorldSpaceIcon.MIN_SIZE
        ## this method is called when the crop handle is moved, onto pixel_coordinate
        if self.corner_idx == 0:
            self.parent.crop_roi[0] = pixel_coordinate[0]
            self.parent.crop_roi[0] = clamp(self.parent.crop_roi[0], 0, self.parent.crop_roi[2] - d)
            self.parent.crop_roi[1] = (self.parent.height - pixel_coordinate[1])
            self.parent.crop_roi[1] = clamp(self.parent.crop_roi[1], 0, self.parent.crop_roi[3] - d)
        elif self.corner_idx == 1:
            self.parent.crop_roi[2] = self.parent.width - (self.parent.width - pixel_coordinate[0])
            self.parent.crop_roi[2] = clamp(self.parent.crop_roi[2], self.parent.crop_roi[0] + d, self.parent.width)
            self.parent.crop_roi[1] = (self.parent.height - pixel_coordinate[1])
            self.parent.crop_roi[1] = clamp(self.parent.crop_roi[1], 0, self.parent.crop_roi[3] - d)
        elif self.corner_idx == 2:
            self.parent.crop_roi[2] = self.parent.width - (self.parent.width - pixel_coordinate[0])
            self.parent.crop_roi[2] = clamp(self.parent.crop_roi[2], self.parent.crop_roi[0] + d, self.parent.width)
            self.parent.crop_roi[3] = (self.parent.height - pixel_coordinate[1])
            self.parent.crop_roi[3] = clamp(self.parent.crop_roi[3], self.parent.crop_roi[1] + d, self.parent.height)
        elif self.corner_idx == 3:
            self.parent.crop_roi[0] = pixel_coordinate[0]
            self.parent.crop_roi[0] = clamp(self.parent.crop_roi[0], 0, self.parent.crop_roi[2] - d)
            self.parent.crop_roi[3] = (self.parent.height - pixel_coordinate[1])
            self.parent.crop_roi[3] = clamp(self.parent.crop_roi[3], self.parent.crop_roi[1] + d, self.parent.height)

    def move_crop_roi(self, dx, dy):
        test_roi = copy(self.parent.crop_roi)
        test_roi[0] += dx
        test_roi[2] += dx
        test_roi[1] += dy
        test_roi[3] += dy

        if test_roi[0] > 0 and test_roi[2] < self.parent.width:
            self.parent.crop_roi[0] = test_roi[0]
            self.parent.crop_roi[2] = test_roi[2]
            self.parent.crop_roi[0] = clamp(self.parent.crop_roi[0], 0, self.parent.width)
            self.parent.crop_roi[2] = clamp(self.parent.crop_roi[2], 0, self.parent.width)
        if test_roi[1] > 0 and test_roi[3] < self.parent.height:
            self.parent.crop_roi[1] = test_roi[1]
            self.parent.crop_roi[3] = test_roi[3]
            self.parent.crop_roi[1] = clamp(self.parent.crop_roi[1], 0, self.parent.height)
            self.parent.crop_roi[3] = clamp(self.parent.crop_roi[3], 0, self.parent.height)

    def convert_crop_roi_to_integers(self):
        for i in range(4):
            self.parent.crop_roi[i] = int(self.parent.crop_roi[i])

    def compute_matrix(self, se_frame, camera):
        self.parent = se_frame
        fpos = se_frame.transform.translation
        fx = [-1, 1, 1, -1]
        fy = [1, 1, -1, -1]
        # find position of the corner of the frame that this handle belongs to
        dx = 0
        dy = 0
        if self.corner_idx == 0:
            dx = se_frame.crop_roi[0]
            dy = -se_frame.crop_roi[1]
        elif self.corner_idx == 1:
            dx = -(se_frame.width - se_frame.crop_roi[2])
            dy = -se_frame.crop_roi[1]
        elif self.corner_idx == 2:
            dx = -(se_frame.width - se_frame.crop_roi[2])
            dy = se_frame.height - se_frame.crop_roi[3]
        elif self.corner_idx == 3:
            dx = se_frame.crop_roi[0]
            dy = se_frame.height - se_frame.crop_roi[3]
        self.transform.translation[0] = fpos[0] + (0.5 * fx[self.corner_idx] * se_frame.width + dx) * se_frame.pixel_size
        self.transform.translation[1] = fpos[1] + (0.5 * fy[self.corner_idx] * se_frame.height + dy) * se_frame.pixel_size
        self.transform.rotation = -self.corner_idx * 90.0
        self.transform.scale = 15.0 / camera.zoom
        self.transform.compute_matrix()

        # set icon's corner positions
        for i in range(4):
            local_corner_pos = tuple(self.corner_positions_local[i])
            vec = np.matrix([*local_corner_pos, 0.0, 1.0]).T
            world_corner_pos = self.transform.matrix * vec
            self.corner_positions[i] = [float(world_corner_pos[0]), float(world_corner_pos[1])]

    def is_hovered(self, camera, cursor_position):
        P = camera.cursor_to_world_position(cursor_position)
        A = self.corner_positions[0]
        B = self.corner_positions[1]
        D = self.corner_positions[3]
        ap = [P[0] - A[0], P[1] - A[1]]
        ab = [B[0] - A[0], B[1] - A[1]]
        ad = [D[0] - A[0], D[1] - A[1]]
        return (0 < ap[0] * ab[0] + ap[1] * ab[1] < ab[0] ** 2 + ab[1] ** 2) and (0 < ap[0] * ad[0] + ap[1] * ad[1] < ad[0] ** 2 + ad[1] ** 2)