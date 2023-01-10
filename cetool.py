import imgui
import config as cfg
from clemframe import *
import time

class CETool:
    title = "Unnamed tool"
    description = "The text entered here is shown as a tooltip when the tool is\n" \
                  "selected and hovered in the Correlation Editor / Tools menu. "

    frames_by_title = list()
    frames_by_title_no_rgb = list()
    FLAG_SHOW_LOCATION_PICKER = False
    selected_position = [0, 0]

    def init(self):
        """Only one CETool exists at a time; when selecting a different type tool
        in the Correlation Editor tools menu, an instance of the selected tool
        object is created and used as the active tool. The CETool class is thus
        fairly minimal, but it does provide some useful wrappers for use in your
        own tool implementations (see the static methods). """
        pass

    def render(self):
        """This is where the GUI of a tool is implemented. Processing may also be
        implemented here, or in on_update. In the programme loop, render() is called
        immediately after on_update"""
        pass


    @staticmethod
    def on_update():
        """Do not override."""
        CETool.frames_by_title = list()
        CETool.frames_by_title_no_rgb = list()
        for f in cfg.ce_frames:
            CETool.frames_by_title.append(f.title)
            if not f.is_rgb:
                CETool.frames_by_title_no_rgb.append(f.title)
        CETool.selected_position = copy(cfg.ce_selected_position)

    @staticmethod
    def widget_select_frame_no_rgb(label, current_frame):
        """Wrapper for an imgui combo (drop-down) menu with all of the current
        frames in the Correlation Editor available for selection.
        :param current_frame: CLEMFrame object
        :return: tuple (bool:changed, CLEMFrame: selected frame)
        """
        current_idx = 0
        if current_frame is not None and current_frame.title in CETool.frames_by_title_no_rgb:
            current_idx = CETool.frames_by_title_no_rgb.index(current_frame.title)
        imgui.text(label)
        _c, idx = imgui.combo("##"+label, current_idx, CETool.frames_by_title_no_rgb)
        selected_frame = cfg.ce_frames[idx] if (idx < len(cfg.ce_frames)) else None
        return _c, selected_frame

    @staticmethod
    def info_selected_position():
        _cw = imgui.get_content_region_available_width()
        imgui.text(f"x: {CETool.selected_position[0]/1000.0:.2f} um")
        imgui.text(f"y: {CETool.selected_position[1]/1000.0:.2f} um")

    @staticmethod
    def centred_button(label, width=None, height=20):
        _cw = imgui.get_content_region_available_width()
        _button_width = width if width else (_cw - 10) / 2
        _button_height = height
        imgui.new_line()
        imgui.same_line(spacing=(_cw - _button_width) / 2.0)
        imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, cfg.CE_WIDGET_ROUNDING)
        clicked = False
        if imgui.button(label, _button_width, _button_height):
            clicked = True
        imgui.pop_style_var()
        return clicked

    @staticmethod
    def tooltip(text):
        if imgui.is_item_hovered():
            if cfg.TOOLTIP_HOVERED_TIMER == 0.0:
                cfg.TOOLTIP_HOVERED_START_TIME = time.time()
                cfg.TOOLTIP_HOVERED_TIMER = 0.001  # add a fake 1 ms to get out of this if clause
            elif cfg.TOOLTIP_HOVERED_TIMER > cfg.TOOLTIP_APPEAR_DELAY:
                imgui.set_tooltip(text)
            else:
                cfg.TOOLTIP_HOVERED_TIMER = time.time() - cfg.TOOLTIP_HOVERED_START_TIME
        if not imgui.is_any_item_hovered():
            cfg.TOOLTIP_HOVERED_TIMER = 0.0
