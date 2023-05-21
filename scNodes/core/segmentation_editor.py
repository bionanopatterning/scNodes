import glfw
import imgui
import scNodes.core.config as cfg
import numpy as np
from itertools import count
import mrcfile
from scNodes.core.opengl_classes import *
import datetime
import scNodes.core.settings as settings
from PIL import Image
from copy import copy
from tkinter import filedialog
import threading
import tifffile
import dill as pickle
from scipy.ndimage import zoom
from scNodes.core.se_models import *


class SegmentationEditor:
    if True:
        CAMERA_ZOOM_STEP = 0.1
        CAMERA_MAX_ZOOM = 100.0
        DEFAULT_HORIZONTAL_FOV_WIDTH = 1000  # upon init, camera zoom is such that from left to right of window = 50 micron.
        DEFAULT_ZOOM = 1.0  # adjusted in init
        DEFAULT_WORLD_PIXEL_SIZE = 1.0  # adjusted on init

        # GUI params
        MAIN_WINDOW_WIDTH = 270
        FEATURE_PANEL_HEIGHT = 104
        INFO_HISTOGRAM_HEIGHT = 70
        SLICER_WINDOW_VERTICAL_OFFSET = 30
        SLICER_WINDOW_WIDTH = 700
        ACTIVE_SLICES_CHILD_HEIGHT = 140
        PROGRESS_BAR_HEIGHT = 8
        MODEL_PANEL_HEIGHT_TRAINING = 158
        MODEL_PANEL_HEIGHT_PREDICTION = 145

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

        self.active_tab = "Segmentation"
        # training dataset params
        self.all_feature_names = list()
        self.trainset_feature_selection = dict()
        self.trainset_selection = list()
        self.trainset_num_boxes_positive = 0
        self.trainset_num_boxes_negative = 0
        self.trainset_boxsize = 32
        self.trainset_apix = 10.0
        self.active_trainset_exports = list()
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

    def set_active_dataset(self, dataset):
        cfg.se_active_frame = dataset
        self.renderer.fbo1 = FrameBuffer(dataset.width, dataset.height, "rgba32f")
        self.renderer.fbo2 = FrameBuffer(dataset.width, dataset.height, "rgba32f")
        self.renderer.fbo3 = FrameBuffer(dataset.width, dataset.height, "rgba32f")
        if dataset.interpolate:
            self.renderer.fbo1.texture.set_linear_interpolation()
            self.renderer.fbo2.texture.set_linear_interpolation()
        else:
            self.renderer.fbo1.texture.set_no_interpolation()
            self.renderer.fbo2.texture.set_no_interpolation()

    def on_update(self):
        imgui.set_current_context(self.imgui_context)
        imgui.CONFIG_DOCKING_ENABLE = True

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

        if self.active_tab == "Models" and cfg.se_active_frame is not None and cfg.se_active_frame.slice_changed:
            for model in cfg.se_models:
                model.set_slice(cfg.se_active_frame.data)
            cfg.se_active_frame.slice_changed = False
        imgui.get_io().display_size = self.window.width, self.window.height
        imgui.new_frame()

        # GUI stuff
        self.camera_control()
        self.camera.on_update()
        self.gui_main()
        self.renderer.render_overlay(self.camera)
        self.input()

        imgui.render()
        self.imgui_implementation.render(imgui.get_draw_data())
        imgui.CONFIG_DOCKING_ENABLE = False

    def input(self):
        if imgui.get_io().want_capture_mouse or imgui.get_io().want_capture_keyboard:
            return

        # key input
        active_frame = cfg.se_active_frame
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
                    elif imgui.is_mouse_down(1):
                        Brush.apply_circular(active_feature, pixel_coordinate, False)
                else:
                    if imgui.is_mouse_clicked(0):
                        active_feature.add_box(pixel_coordinate)
                    elif imgui.is_mouse_clicked(1):
                        active_feature.remove_box(pixel_coordinate)

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
            imgui.push_style_color(imgui.COLOR_CHECK_MARK, *cfg.COLOUR_FRAME_EXTRA_DARK)
            imgui.push_style_color(imgui.COLOR_MENUBAR_BACKGROUND, *cfg.COLOUR_MAIN_MENU_BAR)
            imgui.push_style_color(imgui.COLOR_HEADER, *cfg.COLOUR_HEADER)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *cfg.COLOUR_HEADER_HOVERED)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *cfg.COLOUR_HEADER_ACTIVE)
            imgui.push_style_color(imgui.COLOR_TAB, *cfg.COLOUR_HEADER)
            imgui.push_style_color(imgui.COLOR_TAB_ACTIVE, *cfg.COLOUR_HEADER_ACTIVE)
            imgui.push_style_color(imgui.COLOR_TAB_HOVERED, *cfg.COLOUR_HEADER_HOVERED)
            imgui.push_style_color(imgui.COLOR_DRAG_DROP_TARGET, *cfg.COLOUR_DROP_TARGET)
            imgui.push_style_color(imgui.COLOR_SCROLLBAR_BACKGROUND, *cfg.COLOUR_WINDOW_BACKGROUND)
            imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, cfg.WINDOW_ROUNDING)

        def segmentation_tab():
            def datasets_panel():
                if imgui.collapsing_header("Available datasets", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                    if imgui.begin_child("dataset", 0.0, 80, True, imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR):
                        for s in cfg.se_frames:
                            imgui.push_id(f"se{s.uid}")
                            _change, _selected = imgui.selectable(s.title, cfg.se_active_frame == s)
                            if _change and _selected:
                                self.set_active_dataset(s)
                            if imgui.begin_popup_context_item("##datasetContext"):
                                if imgui.menu_item("Delete dataset")[0]:
                                    cfg.se_frames.remove(s)
                                    if cfg.se_active_frame == s:
                                        cfg.se_active_frame = None
                                imgui.end_popup()
                            imgui.pop_id()
                        imgui.end_child()

            def filters_panel():
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
                        if _minc or _maxc:
                            sef.autocontrast = False
                        imgui.pop_item_width()
                        _c, sef.invert = imgui.checkbox("inverted", sef.invert)
                        imgui.same_line(spacing=21)
                        _c, sef.autocontrast = imgui.checkbox("auto", sef.autocontrast)
                        if _c and sef.autocontrast:
                            sef.requires_histogram_update = True
                        imgui.same_line(spacing=21)
                        _c, sef.interpolate = imgui.checkbox("interpolate", sef.interpolate)
                        if _c:
                            if sef.interpolate:
                                sef.texture.set_linear_interpolation()
                                self.renderer.fbo1.texture.set_linear_interpolation()
                                self.renderer.fbo2.texture.set_linear_interpolation()
                            else:
                                sef.texture.set_no_interpolation()
                                self.renderer.fbo1.texture.set_no_interpolation()
                                self.renderer.fbo2.texture.set_no_interpolation()

                        imgui.separator()
                        fidx = 0
                        for ftr in sef.filters:
                            fidx += 1
                            imgui.push_id(f"filter{fidx}")
                            cw = imgui.get_content_region_available_width()

                            # Filter type selection combo
                            imgui.set_next_item_width(cw - 60)
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (5, 2))
                            _c, ftrtype = imgui.combo("##filtertype", ftr.type, Filter.TYPES)
                            if _c:
                                sef.filters[sef.filters.index(ftr)] = Filter(sef, ftrtype)
                            imgui.same_line()
                            _, ftr.enabled = imgui.checkbox("##enabled", ftr.enabled)
                            if _:
                                ftr.parent.requires_histogram_update = True
                            # Delete button
                            imgui.same_line()
                            if imgui.image_button(self.icon_close.renderer_id, 13, 13):
                                sef.filters.remove(ftr)
                            imgui.pop_style_var(1)
                            # Parameter and strength sliders
                            imgui.push_item_width(cw)
                            if Filter.PARAMETER_NAME[ftr.type] is not None:
                                _c, ftr.param = imgui.slider_float("##param", ftr.param, 0.1, 10.0, format=f"{Filter.PARAMETER_NAME[ftr.type]}: {ftr.param:.1f}")
                                if _c:
                                    ftr.fill_kernel()
                            _, ftr.strength = imgui.slider_float("##strength", ftr.strength, -1.0, 1.0, format=f"weight: {ftr.strength:.2f}")
                            if _:
                                ftr.parent.requires_histogram_update = True
                            imgui.pop_item_width()

                            imgui.pop_id()
                        imgui.set_next_item_width(imgui.get_content_region_available_width())
                        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (5, 2))
                        _c, new_filter_type = imgui.combo("##filtertype", 0, ["Add filter"] + Filter.TYPES)
                        if _c and not new_filter_type == 0:
                            sef.filters.append(Filter(sef, new_filter_type - 1))
                        imgui.pop_style_var(6)
                        imgui.pop_style_color(1)

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
                        if imgui.begin_child(f"##{f.uid}", 0.0, SegmentationEditor.FEATURE_PANEL_HEIGHT, True):
                            cw = imgui.get_content_region_available_width()

                            # Colour picker
                            _, f.colour = imgui.color_edit3(f.title, *f.colour[:3],
                                                            imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)

                            # Title
                            imgui.same_line()
                            imgui.set_next_item_width(cw - 25)
                            _, f.title = imgui.input_text("##title", f.title, 256, imgui.INPUT_TEXT_NO_HORIZONTAL_SCROLL | imgui.INPUT_TEXT_AUTO_SELECT_ALL)

                            # Alpha slider and brush size
                            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                            imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                            imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                            imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                            imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                            imgui.set_next_item_width(cw - 40)
                            pxs = cfg.se_active_frame.pixel_size
                            _, f.brush_size = imgui.slider_float("brush", f.brush_size, 1.0, 25.0 / pxs / 2,
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
                                imgui.text("Active slices")
                                if imgui.begin_child("active_slices", 180, SegmentationEditor.ACTIVE_SLICES_CHILD_HEIGHT, True):
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
                    imgui.pop_style_var(1)

            datasets_panel()
            filters_panel()
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
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 2))

                imgui.text("Features of interest")
                if imgui.begin_child("features", 0.0, 1 + len(self.all_feature_names) * 20, False, imgui.WINDOW_NO_SCROLLBAR):
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
                if imgui.begin_child("datasets", 0.0, min([80, 10 + len(cfg.se_frames)*20]), True):
                    for s in cfg.se_frames:
                        imgui.push_id(f"{s.uid}")
                        _, s.sample = imgui.checkbox(s.title, s.sample)
                        imgui.pop_id()
                    imgui.end_child()
                imgui.pop_style_color()


                imgui.text("Set parameters")
                if imgui.begin_child("params", 0.0, 80, True):
                    imgui.push_item_width(cw - 53)
                    _, self.trainset_boxsize = imgui.slider_int("boxes", self.trainset_boxsize, 8, 128, format=f"{self.trainset_boxsize} pixel")
                    _, self.trainset_apix = imgui.slider_float("A/pix", self.trainset_apix, 1.0, 20.0, format=f"{self.trainset_apix:.2f}")
                    imgui.pop_item_width()
                    calculate_number_of_boxes()
                    imgui.text(f"Positive samples: {self.trainset_num_boxes_positive}")
                    imgui.text(f"Negative samples: {self.trainset_num_boxes_negative}")
                    imgui.end_child()

                # 'Generate set' button
                cw = imgui.get_content_region_available_width()

                for process in self.active_trainset_exports:
                    SegmentationEditor._gui_background_process_progress_bar(process)
                    if process.progress == 1.0:
                        self.active_trainset_exports.remove(process)
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.new_line()
                imgui.same_line(spacing=(cw - 120) / 2)
                if imgui.button("Generate set", 120, 23):
                    self.launch_create_training_set()
                imgui.pop_style_var(6)

            if imgui.collapsing_header("Available datasets", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                if imgui.begin_child("dataset", 0.0, 80, True, imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR):
                    for s in cfg.se_frames:
                        imgui.push_id(f"se{s.uid}")
                        _change, _selected = imgui.selectable(s.title, cfg.se_active_frame == s)
                        if _change and _selected:
                            self.set_active_dataset(s)
                        if imgui.begin_popup_context_item("##datasetContext"):
                            if imgui.menu_item("Delete dataset")[0]:
                                cfg.se_frames.remove(s)
                                if cfg.se_active_frame == s:
                                    cfg.se_active_frame = None
                            imgui.end_popup()
                        imgui.pop_id()
                    imgui.end_child()

            if imgui.collapsing_header("Models", None, imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                for m in cfg.se_models:
                    pop_active_colour = False
                    if cfg.se_active_model == m:
                        pop_active_colour = True
                        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *cfg.COLOUR_FRAME_ACTIVE)

                    panel_height = SegmentationEditor.MODEL_PANEL_HEIGHT_TRAINING if m.active_tab == 0 else SegmentationEditor.MODEL_PANEL_HEIGHT_PREDICTION
                    panel_height += 10 if m.background_process is not None else 0
                    if imgui.begin_child(f"SEModel_{m.uid}", 0.0, panel_height, True, imgui.WINDOW_NO_SCROLLBAR):
                        cw = imgui.get_content_region_available_width()

                        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
                        if m.background_process is None:
                            # delete button
                            if imgui.image_button(self.icon_close.renderer_id, 19, 19):
                                cfg.se_models.remove(m)
                                if cfg.se_active_model == m:
                                    cfg.se_active_model = None
                                m.delete()
                        else:
                            if imgui.image_button(self.icon_stop.renderer_id, 19, 19):
                                m.background_process.stop()

                        imgui.pop_style_var(1)

                        imgui.same_line()
                        _, m.colour = imgui.color_edit3(m.title, *m.colour[:3], imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_TOOLTIP | imgui.COLOR_EDIT_NO_DRAG_DROP)
                        # Title
                        imgui.same_line()
                        imgui.set_next_item_width(cw - 55)
                        _, m.title = imgui.input_text("##title", m.title, 256, imgui.INPUT_TEXT_NO_HORIZONTAL_SCROLL | imgui.INPUT_TEXT_AUTO_SELECT_ALL)

                        # Model selection
                        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (5, 2))
                        imgui.align_text_to_frame_padding()
                        if m.compiled:
                            model_info = SEModel.AVAILABLE_MODELS[m.model_enum]+f" ({m.n_parameters}, {m.box_size}, {m.apix:.3f})"
                            imgui.text(model_info)
                        else:
                            imgui.text("Model:")
                            imgui.same_line()
                            imgui.set_next_item_width(cw - 51)
                            _, m.model_enum = imgui.combo("##model_type", m.model_enum, SEModel.AVAILABLE_MODELS)
                        imgui.pop_style_var()

                        if imgui.begin_tab_bar("##tabs"):
                            if imgui.begin_tab_item("   Training   ")[0]:
                                m.active_tab = 0
                                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (3, 3))
                                # Train data selection
                                cw = imgui.get_content_region_available_width()
                                imgui.set_next_item_width(cw - 65)
                                _, m.train_data_path = imgui.input_text("##training_data", m.train_data_path, 256)
                                imgui.pop_style_var()
                                imgui.same_line()
                                if imgui.button("browse", 55, 19):
                                    m.train_data_path = filedialog.askopenfilename(filetypes=[("scNodes traindata", f"{cfg.filetype_traindata}")])

                                # Training parameters
                                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                                imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                                imgui.push_style_var(imgui.STYLE_GRAB_MIN_SIZE, 9)
                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))

                                imgui.push_item_width(cw)
                                _, m.epochs = imgui.slider_int("##epochs", m.epochs, 1, 100, f"{m.epochs} epoch"+("s" if m.epochs>1 else ""))
                                _, m.batch_size = imgui.slider_int("##batchs", m.batch_size, 1, 128, f"{m.batch_size} batch size")
                                imgui.pop_item_width()
                                imgui.pop_style_var(1)
                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 2))

                                # Load, save, train buttons.
                                block_buttons = False
                                if m.background_process is not None:
                                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_NEUTRAL_LIGHT)
                                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *cfg.COLOUR_NEUTRAL_LIGHT)
                                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *cfg.COLOUR_NEUTRAL_LIGHT)
                                    imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_NEUTRAL)
                                    block_buttons = True
                                if imgui.button("load", (cw - 16) / 3, 20):
                                    if not block_buttons:
                                        model_path = filedialog.askopenfilename(filetypes=[("scNodes CNN", f"{cfg.filetype_semodel}")])
                                        if model_path is not None:
                                            m.load(model_path)
                                imgui.same_line(spacing=8)
                                block_save_button = False
                                if m.model is None:
                                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *cfg.COLOUR_NEUTRAL_LIGHT)
                                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED,*cfg.COLOUR_NEUTRAL_LIGHT)
                                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE,*cfg.COLOUR_NEUTRAL_LIGHT)
                                    imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_NEUTRAL)
                                    block_save_button = True
                                if imgui.button("save", (cw - 16) / 3, 20):
                                    if not block_buttons or block_save_button:
                                        model_path = filedialog.asksaveasfilename()
                                        if model_path is not None:
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

                            if imgui.begin_tab_item("    Prediction   ")[0]:
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
                                _, m.overlap = imgui.slider_float("##overlap", m.overlap, 0.0, 0.5, format=f"{m.overlap:.2f} overlap")
                                _, m.threshold = imgui.slider_float("##thershold", m.threshold, 0.0, 1.0, format=f"{m.threshold:.2f} threshold")
                                imgui.pop_item_width()

                                _, m.active = imgui.checkbox("active    ", m.active)
                                if _ and m.active:
                                    m.set_slice(cfg.se_active_frame.data)
                                imgui.same_line()
                                _, m.blend = imgui.checkbox("blend    ", m.blend)
                                imgui.same_line()
                                _, m.show = imgui.checkbox("show    ", m.show)
                                imgui.pop_style_var(5)
                                imgui.pop_style_color(1)
                                imgui.end_tab_item()
                            imgui.end_tab_bar()

                        if m.background_process is not None:
                            SegmentationEditor._gui_background_process_progress_bar(m.background_process)
                            if m.background_process.progress >= 1.0:
                                m.background_process = None
                        imgui.end_child()
                    if pop_active_colour:
                        imgui.pop_style_color(1)

                cw = imgui.get_content_region_available_width()
                imgui.new_line()
                imgui.same_line(spacing=(cw - 120) / 2)
                imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, 10)
                imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, 20)
                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 1)
                if imgui.button("Add model", 120, 23):
                    cfg.se_models.append(SEModel())
                imgui.pop_style_var(3)

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
                                if filename[-4:] == ".mrc":
                                    cfg.se_frames.append(SEFrame(filename))
                                    self.set_active_dataset(cfg.se_frames[-1])
                                elif filename[-5:] == ".scns":
                                    with open(filename, 'rb') as pickle_file:
                                        seframe = pickle.load(pickle_file)
                                        seframe.on_load()
                                        seframe.slice_changed = False
                                        cfg.se_frames.append(seframe)
                                        self.set_active_dataset(cfg.se_frames[-1])
                            self.parse_available_features()
                        except Exception as e:
                            print(e)
                    if imgui.menu_item("Save dataset")[0]:
                        try:
                            filename = filedialog.asksaveasfilename(filetypes=[("scNodes segmentation", ".scns")])
                            if filename != '':
                                if filename[-5:] != ".scns":
                                    filename += ".scns"
                                with open(filename, 'wb') as pickle_file:
                                    pickle.dump(cfg.se_active_frame, pickle_file)
                        except Exception as e:
                            print(e)

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
            imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *cfg.COLOUR_TITLE_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *cfg.COLOUR_TITLE_BACKGROUND)
            imgui.set_next_window_size(SegmentationEditor.SLICER_WINDOW_WIDTH, 0.0)
            window_x_pos = SegmentationEditor.MAIN_WINDOW_WIDTH + (self.window.width - SegmentationEditor.MAIN_WINDOW_WIDTH - SegmentationEditor.SLICER_WINDOW_WIDTH) / 2
            imgui.set_next_window_position(window_x_pos, self.window.height - SegmentationEditor.SLICER_WINDOW_VERTICAL_OFFSET)
            imgui.begin("##slicer", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_BACKGROUND)
            imgui.set_next_item_width(imgui.get_content_region_available_width())
            frame = cfg.se_active_frame
            slice_idx_changed, requested_slice = imgui.slider_int("##slicer_slider", frame.current_slice, 0, frame.n_slices, format=f"slice {1+frame.current_slice}/{frame.n_slices}")
            if slice_idx_changed:
                frame.set_slice(requested_slice)
            imgui.end()

            imgui.pop_style_var(3)
            imgui.pop_style_color(5)

        # START GUI:
        # Menu bar
        menu_bar()
        # Render the currently active frame

        if cfg.se_active_frame is not None:
            if self.active_tab == "Segmentation":
                pxd = self.renderer.render_frame_with_segmentations(cfg.se_active_frame, self.camera, self.window, cfg.se_active_frame.filters)
                if pxd is not None:
                    cfg.se_active_frame.compute_histogram(pxd)
                    if cfg.se_active_frame.autocontrast:
                        cfg.se_active_frame.compute_autocontrast(None, pxd)
                    cfg.se_active_frame.requires_histogram_update = False

                # render drawing ROI indicator
                active_feature = cfg.se_active_frame.active_feature
                if active_feature is not None:
                    radius = active_feature.brush_size * active_feature.parent.pixel_size
                    world_position = self.camera.cursor_to_world_position(self.window.cursor_pos)
                    if not imgui.is_key_down(glfw.KEY_LEFT_SHIFT):
                        self.renderer.add_circle(world_position, radius, active_feature.colour)
                    else:
                        self.renderer.add_square(world_position, active_feature.box_size_nm, active_feature.colour)
                for f in cfg.se_active_frame.features:
                    frame_xy = f.parent.transform.translation
                    if f.show_boxes and not f.hide:
                        for box in f.boxes[f.current_slice]:
                            box_x_pos = frame_xy[0] + (box[0] - f.parent.width / 2) * f.parent.pixel_size
                            box_y_pos = frame_xy[1] + (box[1] - f.parent.height / 2) * f.parent.pixel_size
                            self.renderer.add_square((box_x_pos, box_y_pos), f.box_size_nm, f.colour)

            if self.active_tab == "Models":
                self.renderer.render_frame_with_models(cfg.se_active_frame, self.camera)

        # MAIN WINDOW
        imgui.set_next_window_position(0, 17, imgui.ONCE)
        imgui.set_next_window_size(SegmentationEditor.MAIN_WINDOW_WIDTH, self.window.height - 17)

        imgui.begin("##se_main", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        if imgui.begin_tab_bar("##tabs"):
            if imgui.begin_tab_item("Segmentation")[0]:
                segmentation_tab()
                self.active_tab = "Segmentation"
                imgui.end_tab_item()
            if imgui.begin_tab_item("Models")[0]:
                if self.active_tab != "Models":
                    self.parse_available_features()
                self.active_tab = "Models"
                models_tab()
                imgui.end_tab_item()
            imgui.end_tab_bar()

        imgui.end()

        slicer_window()

        imgui.pop_style_color(32)
        imgui.pop_style_var(1)

    def parse_available_features(self):
        # upon opening Models tab.
        self.all_feature_names = list()
        for sef in cfg.se_frames:
            for ftr in sef.features:
                if ftr.title not in self.all_feature_names:
                    self.all_feature_names.append(ftr.title)
                if ftr.title not in self.trainset_feature_selection:
                    self.trainset_feature_selection[ftr.title] = 0.0
        to_pop = list()
        for key in self.trainset_feature_selection.keys():
            if key not in self.all_feature_names:
                to_pop.append(key)
        for key in to_pop:
            self.trainset_feature_selection.pop(key)

    @staticmethod
    def _gui_background_process_progress_bar(process):
        cw = imgui.get_content_region_available_width()
        origin = imgui.get_window_position()
        y = imgui.get_cursor_screen_pos()[1]
        drawlist = imgui.get_window_draw_list()
        drawlist.add_rect_filled(8 + origin[0], y,
                                 8 + origin[0] + cw,
                                 y + SegmentationEditor.PROGRESS_BAR_HEIGHT,
                                 imgui.get_color_u32_rgba(*cfg.COLOUR_NEUTRAL))
        drawlist.add_rect_filled(8 + origin[0], y,
                                 8 + origin[0] + cw * min([1.0, process.progress]),
                                 y + SegmentationEditor.PROGRESS_BAR_HEIGHT,
                                 imgui.get_color_u32_rgba(*cfg.COLOUR_POSITIVE))
        imgui.dummy(0, SegmentationEditor.PROGRESS_BAR_HEIGHT)

    def launch_create_training_set(self):
        path = filedialog.asksaveasfilename()
        if path is None:
            return
        path += cfg.filetype_traindata
        #
        positive_feature_names = list()
        negative_feature_names = list()
        for f in self.all_feature_names:
            if self.trainset_feature_selection[f] == 1:
                positive_feature_names.append(f)
            elif self.trainset_feature_selection[f] == -1:
                negative_feature_names.append(f)

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
        target_type_dict = {np.float32: float, float: float, np.int8: np.uint8, np.int16: np.uint16}

        crop_nm = boxsize * apix
        for d in datasets:
            mrcf = mrcfile.mmap(d.path, mode="r")
            raw_type = type(mrcf.data[0, 0, 0])
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
                    pass
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
                                segmentation = np.flipud(f.slices[z][y_min:y_max, x_min:x_max])
                                # scale image to the requested pixel size
                                image = zoom(image, scale_fac)
                                segmentation = zoom(segmentation, scale_fac)
                                image = image[:boxsize, :boxsize]
                                segmentation = segmentation[:boxsize, :boxsize]
                                if is_positive:
                                    positive.append([image, segmentation])
                                else:
                                    negative.append([image, np.zeros_like(segmentation)])
                            n_done += 1
                            process.set_progress(n_done / n_boxes)

        all_imgs = np.array(positive + negative)
        tifffile.imwrite(path, all_imgs, description=f"apix={apix}")

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

    HISTOGRAM_BINS = 40

    def __init__(self, path):
        uid_counter = next(SEFrame.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')+"000") + uid_counter
        self.path = path
        self.title = os.path.splitext(os.path.basename(self.path))[0]
        self.n_slices = 0
        self.current_slice = -1
        self.slice_changed = False
        self.data = None
        self.features = list()
        self.active_feature = None
        self.height, self.width = mrcfile.mmap(self.path, mode="r").data.shape[1:3]
        self.pixel_size = mrcfile.open(self.path, header_only=True).voxel_size.x / 10.0
        self.title += f" ({self.pixel_size * 10.0:.2f} A/pix)"
        self.transform = Transform()
        self.texture = None
        self.quad_va = None
        self.border_va = None
        self.interpolate = False
        self.alpha = 1.0
        self.filters = list()
        self.invert = True
        self.autocontrast = True
        self.sample = True
        self.hist_vals = list()
        self.hist_bins = list()
        self.requires_histogram_update = False
        self.corner_positions_local = []
        self.set_slice(0, False)
        self.setup_opengl_objects()
        self.contrast_lims = [0, 65535.0]
        self.compute_autocontrast()
        self.compute_histogram()

        self.features.append(Segmentation(self, "Unnamed feature"))
        self.active_feature = self.features[-1]

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
        self.requires_histogram_update = True
        if requested_slice == self.current_slice:
            return
        self.slice_changed = True
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

    def compute_autocontrast(self, saturation=None, pxd=None):
        data = self.data if pxd is None else pxd
        #s = data.shape
        #data = data[int(s[0]/3):int(s[0]*2/3), int(s[1]/3):int(s[1]*2/3)]
        saturation_pct = settings.autocontrast_saturation
        if saturation:
            saturation_pct = saturation
        subsample = data[::settings.autocontrast_subsample, ::settings.autocontrast_subsample]
        n = subsample.shape[0] * subsample.shape[1]
        sorted_pixelvals = np.sort(subsample.flatten())

        min_idx = min([int(saturation_pct / 100.0 * n), n - 1])
        max_idx = max([int((1.0 - saturation_pct / 100.0) * n), 0])
        self.contrast_lims[0] = sorted_pixelvals[min_idx]
        self.contrast_lims[1] = sorted_pixelvals[max_idx]

    def compute_histogram(self, pxd=None):
        self.requires_histogram_update = False
        data = self.data if pxd is None else pxd
        # ignore very bright pixels
        mean = np.mean(data)
        std = np.std(data)
        self.hist_vals, self.hist_bins = np.histogram(data[data < (mean + 20 * std)], bins=SEFrame.HISTOGRAM_BINS)

        self.hist_vals = self.hist_vals.astype('float32')
        self.hist_bins = self.hist_bins.astype('float32')
        self.hist_vals = np.log(self.hist_vals + 1)

    def on_load(self):
        uid_counter = next(SEFrame.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        self.setup_opengl_objects()
        for f in self.features:
            f.on_load()
        for f in self.filters:
            f.fill_kernel()

    @staticmethod
    def from_clemframe(clemframe):
        se_frame = SEFrame(clemframe.path)


    def __eq__(self, other):
        if isinstance(other, SEFrame):
            return self.uid == other.uid
        return False


class Segmentation:
    idgen = count(0)

    DEFAULT_COLOURS = [(66 / 255, 214 / 255, 164 / 255),
                       (255 / 255, 243 / 255, 0 / 255),
                       (255 / 255, 104 / 255, 0 / 255),
                       (255 / 255, 13 / 255, 0 / 255),
                       (174 / 255, 0 / 255, 255 / 255),
                       (21 / 255, 0 / 255, 255 / 255),
                       (0 / 255, 136 / 255, 266 / 255),
                       (0 / 255, 247 / 255, 255 / 255),
                       (0 / 255, 255 / 255, 0 / 255)]


    def __init__(self, parent_frame, title):
        uid_counter = next(Segmentation.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        self.parent = parent_frame
        self.parent.active_feature = self
        self.width = self.parent.width
        self.height = self.parent.height
        self.title = title
        self.colour = Segmentation.DEFAULT_COLOURS[(len(self.parent.features)) % len(Segmentation.DEFAULT_COLOURS)]
        self.alpha = 0.33
        self.hide = False
        self.contour = False
        self.expanded = False
        self.brush_size = 10.0
        self.show_boxes = True
        self.box_size = 32
        self.box_size_nm = self.box_size * self.parent.pixel_size
        self.slices = dict()
        self.boxes = dict()
        self.n_boxes = 0
        self.edited_slices = list()
        self.current_slice = -1
        self.data = None
        self.texture = Texture(format="r32f")
        self.texture.update(None, self.width, self.height)
        self.texture.set_linear_interpolation()
        self.set_slice(self.parent.current_slice)

    def on_load(self):
        uid_counter = next(Segmentation.idgen)
        self.uid = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + "000") + uid_counter
        self.texture = Texture(format="r32f")
        cslice = self.current_slice
        self.current_slice = -1
        self.set_slice(cslice)

    def set_box_size(self, box_size_px):
        self.box_size = box_size_px
        self.box_size_nm = self.box_size * self.parent.pixel_size

    def add_box(self, pixel_coordinates):
        if len(self.boxes[self.current_slice]) == 0:
            self.request_draw_in_current_slice()
        self.boxes[self.current_slice].append(pixel_coordinates)
        self.n_boxes += 1

    def remove_box(self, pixel_coordinate):
        box_list = self.boxes[self.current_slice]
        x = pixel_coordinate[0]
        y = pixel_coordinate[1]
        d = np.inf
        idx = None
        for i in range(len(box_list)):
            _d = (box_list[i][0]-x)**2 + (box_list[i][1]-y)**2
            if _d < d:
                d = _d
                idx = i
        if idx is not None:
            self.boxes[self.current_slice].pop(idx)
            self.n_boxes -= 1

    def set_slice(self, requested_slice):
        if requested_slice == self.current_slice:
            return
        self.current_slice = requested_slice
        if requested_slice in self.slices:
            self.data = self.slices[requested_slice]
            if self.data is None:
                self.texture.update(self.data, self.width, self.height)
            else:
                self.texture.update(self.data)
        else:
            self.boxes[requested_slice] = list()
            self.slices[requested_slice] = None
            self.data = None
            self.texture.update(self.data, self.width, self.height)

    def remove_slice(self, requested_slice):
        if requested_slice in self.edited_slices:
            self.edited_slices.remove(requested_slice)
            self.slices.pop(requested_slice)
            if self.current_slice == requested_slice:
                self.data *= 0
                self.texture.update(self.data, self.width, self.height)
        if requested_slice in self.boxes:
            self.boxes[requested_slice] = list()

    def request_draw_in_current_slice(self):
        if self.current_slice in self.slices:
            if self.slices[self.current_slice] is None:
                self.slices[self.current_slice] = np.zeros((self.height, self.width), dtype=np.uint8)
                self.data = self.slices[self.current_slice]
                self.texture.update(self.data, self.width, self.height)
                self.edited_slices.append(self.current_slice)
        else:
            self.slices[self.current_slice] = np.zeros((self.height, self.width), dtype=np.uint8)
            self.data = self.slices[self.current_slice]
            self.texture.update(self.data, self.width, self.height)
            self.edited_slices.append(self.current_slice)


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


class Filter:

    TYPES = ["Gaussian blur", "Box blur", "Sobel vertical", "Sobel horizontal"]
    PARAMETER_NAME = ["sigma", "box", None, None]
    M = 16

    def __init__(self, parent, filter_type):
        self.parent = parent
        self.type = filter_type  # integer, corresponding to an index in the Filter.TYPES list
        self.k1 = np.zeros((Filter.M*2+1, 1), dtype=np.float32)
        self.k2 = np.zeros((Filter.M * 2 + 1, 1), dtype=np.float32)
        self.enabled = True
        self.ssbo1 = -1
        self.ssbo2 = -1
        self.param = 1.0
        self.strength = 1.0
        self.fill_kernel()

    def upload_buffer(self):
        if self.ssbo1 != -1:
            glDeleteBuffers(2, [self.ssbo1, self.ssbo2])
        self.ssbo1 = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo1)
        glBufferData(GL_SHADER_STORAGE_BUFFER, (Filter.M * 2 + 1) * 4, self.k1.flatten(), GL_STATIC_READ)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        self.ssbo2 = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo2)
        glBufferData(GL_SHADER_STORAGE_BUFFER, (Filter.M * 2 + 1) * 4, self.k2.flatten(), GL_STATIC_READ)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)


    def bind(self, horizontal=True):
        if horizontal:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo1)
        else:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo2)

    def unbind(self):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def fill_kernel(self):
        if self.type == 0:
            self.k1 = np.exp(-np.linspace(-Filter.M, Filter.M, 2*Filter.M+1)**2 / self.param**2, dtype=np.float32)
            self.k1 /= np.sum(self.k1, dtype=np.float32)
            self.k2 = self.k1
        if self.type == 1:
            m = min([int(self.param), 7])
            self.k1 = np.asarray([0] * (Filter.M - 1 - m) + [1] * m + [1] + [1] * m + [0] * (Filter.M - 1 - m)) / (2 * m + 1)
            self.k2 = self.k1
        if self.type == 2:
            self.k1 = np.asarray([0] * (Filter.M - 1) + [1, 2, 1] + [0] * (Filter.M - 1))
            self.k2 = np.asarray([0] * (Filter.M - 1) + [1, 0, -1] + [0] * (Filter.M - 1))
        if self.type == 3:
            self.k1 = np.asarray([0] * (Filter.M - 1) + [1, 0, -1] + [0] * (Filter.M - 1))
            self.k2 = np.asarray([0] * (Filter.M - 1) + [1, 2, 1] + [0] * (Filter.M - 1))

        self.k1 = np.asarray(self.k1, dtype=np.float32)
        self.k2 = np.asarray(self.k2, dtype=np.float32)
        self.upload_buffer()

        self.parent.requires_histogram_update = True


class Renderer:
    def __init__(self):
        self.quad_shader = Shader(os.path.join(cfg.root, "shaders", "se_quad_shader.glsl"))
        self.b_segmentation_shader = Shader(os.path.join(cfg.root, "shaders", "se_binary_segmentation_shader.glsl"))
        self.f_segmentation_shader = Shader(os.path.join(cfg.root, "shaders", "se_float_segmentation_shader.glsl"))
        self.border_shader = Shader(os.path.join(cfg.root, "shaders", "se_border_shader.glsl"))
        self.kernel_filter = Shader(os.path.join(cfg.root, "shaders", "se_compute_kernel_filter.glsl"))
        self.mix_filtered = Shader(os.path.join(cfg.root, "shaders", "se_compute_mix.glsl"))
        self.line_shader = Shader(os.path.join(cfg.root, "shaders", "ce_line_shader.glsl"))
        self.line_list = list()
        self.line_list_s = list()
        self.line_va = VertexArray(None, None, attribute_format="xyrgb")
        self.fbo1 = FrameBuffer()
        self.fbo2 = FrameBuffer()
        self.fbo3 = FrameBuffer()

    def render_frame_with_models(self, se_frame, camera):
        se_frame.update_model_matrix()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)

        self.quad_shader.bind()
        se_frame.quad_va.bind()
        se_frame.texture.bind()
        self.quad_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.quad_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        self.quad_shader.uniform1f("alpha", se_frame.alpha)
        self.quad_shader.uniform1f("contrastMin", se_frame.contrast_lims[0])
        self.quad_shader.uniform1f("contrastMax", se_frame.contrast_lims[1])
        glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.quad_shader.unbind()
        se_frame.quad_va.unbind()
        glActiveTexture(GL_TEXTURE0)

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
        glDrawElements(GL_LINES, se_frame.border_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.border_shader.unbind()
        se_frame.border_va.unbind()

    def render_frame_with_segmentations(self, se_frame, camera, window, filters=[]):
        """
        se_frame: the SEFrame object to render.
        camera: a Camera object to render with.
        window: the Window object (to which the viewport size will be reset)
        filters: a list of Filter objectm, to apply to the se_frame pixeldata.
        """
        se_frame.update_model_matrix()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)

        self.fbo1.clear((0.0, 0.0, 0.0, 1.0))
        self.fbo2.clear((0.0, 0.0, 0.0, 1.0))
        self.fbo3.clear((0.0, 0.0, 0.0, 2.0))

        # render the image to a framebuffer
        fake_camera_matrix = np.matrix([[2 / self.fbo1.width, 0, 0, 0], [0, 2 / self.fbo1.height, 0, 0], [0, 0, -2 / 100, 0], [0, 0, 0, 1]])
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
        self.quad_shader.bind()
        se_frame.quad_va.bind()
        self.fbo1.texture.bind(0)
        self.quad_shader.uniformmat4("cameraMatrix", camera.view_projection_matrix)
        self.quad_shader.uniformmat4("modelMatrix", se_frame.transform.matrix)
        self.quad_shader.uniform1f("alpha", se_frame.alpha)
        if se_frame.invert:
            self.quad_shader.uniform1f("contrastMin", se_frame.contrast_lims[1])
            self.quad_shader.uniform1f("contrastMax", se_frame.contrast_lims[0])
        else:
            self.quad_shader.uniform1f("contrastMin", se_frame.contrast_lims[0])
            self.quad_shader.uniform1f("contrastMax", se_frame.contrast_lims[1])
        glDrawElements(GL_TRIANGLES, se_frame.quad_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.quad_shader.unbind()
        se_frame.quad_va.unbind()
        glActiveTexture(GL_TEXTURE0)

        # render overlays
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
        glDrawElements(GL_LINES, se_frame.border_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.border_shader.unbind()
        se_frame.border_va.unbind()

        return pxd

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

    def render_overlay(self, camera):
        self.render_lines(camera)


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


class BackgroundProcess:
    idgen = count(0)

    def __init__(self, function, args, name=None):
        self.uid = next(BackgroundProcess.idgen)
        self.function = function
        self.args = args
        self.name = name
        self.thread = None
        self.progress = 0.0

    def start(self):
        _name = f"BackgroundProcess {self.uid} - "+(self.name if self.name is not None else "")
        self.thread = threading.Thread(daemon=True, target=self._run, name=_name)
        self.thread.start()

    def _run(self):
        self.function(*self.args, self)

    def set_progress(self, progress):
        self.progress = progress

    def __str__(self):
        return f"BackgroundProcess {self.uid} with function {self.function} and args {self.args}"
