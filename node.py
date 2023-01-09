from itertools import count
import config as cfg
import imgui
import time
import datetime
import copy
import numpy as np
from dataset import *
import settings
import dill as pickle
from joblib import Parallel, delayed


class Node:
    title = "NullNode"
    group = "Ungrouped"
    colour = (1.0, 1.0, 1.0, 1.0)
    id_generator = count(0)
    size = 200
    sortid = 9999

    COLOUR_WINDOW_BACKGROUND = (0.96, 0.96, 0.96, 0.96)
    COLOUR_FOCUSED_NODE_WINDOW_BACKGROUND = (0.99, 0.93, 0.93, 0.96)
    COLOUR_WINDOW_BORDER = (0.45, 0.45, 0.45, 1.0)
    COLOUR_WINDOW_BORDER_ACTIVE_NODE = (0.25, 0.25, 0.25, 1.0)
    COLOUR_WINDOW_BORDER_FOCUSED_NODE = (0 / 255, 0 / 255, 0 / 255)
    ACTIVE_NODE_BORDER_THICKNESS = 1.3
    FOCUSED_NODE_BORDER_THICKNESS = 1.1
    DEFAULT_NODE_BORDER_THICKNESS = 1.0
    COLOUR_TEXT = (0.0, 0.0, 0.0, 1.0)
    COLOUR_TEXT_DISABLED = (0.2, 0.4, 0.2, 1.0)
    COLOUR_FRAME_BACKGROUND = (0.84, 0.84, 0.84, 1.0)

    WINDOW_ROUNDING = 5.0
    FRAME_ROUNDING = 2.0
    PLAY_BUTTON_SIZE = 40
    PLAY_BUTTON_ICON_SIZE = 5.0
    PLAY_BUTTON_ICON_LINEWIDTH = 12.0
    PLAY_BUTTON_ICON_COLOUR = (0.2, 0.8, 0.2, 1.0)
    PLAY_BUTTON_WINDOW_BACKGROUND = (0.96, 0.96, 0.96, 1.0)
    PROGRESS_BAR_HEIGHT = 10
    PROGRESS_BAR_PADDING = 8

    TOOLTIP_APPEAR_DELAY = 1.0  # seconds
    TOOLTIP_HOVERED_TIMER = 0.0
    TOOLTIP_HOVERED_START_TIME = 0.0

    DISABLE_FRAME_INFO_WINDOW = False

    def __init__(self):
        self.id = int(datetime.datetime.now().strftime("%Y%m%d")+"000") + next(Node.id_generator)
        self.position = [0, 0]
        self.last_measured_window_position = [0, 0]
        self.node_height = 100
        self.connectable_attributes = list()
        self.play = False
        self.any_change = False
        self.queued_actions = list()
        self.use_roi = False
        self.roi = [0, 0, 0, 0]
        self.lut = "auto"
        cfg.nodes.append(self)

        # some bookkeeping vars - ignore these
        self.buffer_last_output = False
        self.last_index_requested = -1
        self.last_frame_returned = None

        self.keep_active = False
        self.does_profiling_time = True
        self.does_profiling_count = True
        self.profiler_time = 0.0
        self.profiler_count = 0

        # flags
        self.NODE_RETURNS_IMAGE = True
        self.NODE_IS_DATA_SOURCE = False
        self.ROI_MUST_BE_SQUARE = False
        self.NODE_GAINED_FOCUS = False
        self.FRAME_REQUESTED_BY_IMAGE_VIEWER = False  # bit of a misc one. when the image viewer is requesting a frame (rather than the processing pipeline asking for one), this flag is temporarily set to True. This is used in e.g. the CropImage node, where the output to the image viewer is different than the output to the next node.
        self.OVERRIDE_AUTOCONTRAST = False
        self.OVERRIDE_AUTOCONTRAST_LIMS = (0, 65535)


    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False

    def clear_flags(self):
        for a in self.connectable_attributes:
            a.clear_flags()
        if self.any_change and not cfg.focused_node == self:
            cfg.any_change = True
        self.any_change = False
        self.NODE_GAINED_FOCUS = False

    def render_start(self):
        if self.use_roi == False:
            self.roi = [0, 0, 0, 0]
        ## render the node window
        imgui.set_next_window_position(self.position[0], self.position[1], imgui.ALWAYS)
        imgui.set_next_window_size_constraints((self.size, 50), (self.size, 1000))
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *self.colour)
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, *self.colour)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *self.colour)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *self.colour)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *self.colour)
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *self.colour)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *self.colour)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *self.colour_brighten(self.colour))
        if cfg.focused_node == self:
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *self.colour_whiten(self.colour))
            imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, Node.FOCUSED_NODE_BORDER_THICKNESS)
            imgui.push_style_color(imgui.COLOR_BORDER, *Node.COLOUR_WINDOW_BORDER_FOCUSED_NODE)
            imgui.push_style_color(imgui.COLOR_BORDER_SHADOW, *Node.COLOUR_WINDOW_BORDER_FOCUSED_NODE)
        elif cfg.active_node == self:
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *Node.COLOUR_WINDOW_BACKGROUND)
            imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, Node.ACTIVE_NODE_BORDER_THICKNESS)
            imgui.push_style_color(imgui.COLOR_BORDER, *Node.COLOUR_WINDOW_BORDER_ACTIVE_NODE)
            imgui.push_style_color(imgui.COLOR_BORDER_SHADOW, *Node.COLOUR_WINDOW_BORDER_ACTIVE_NODE)
        else:
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *Node.COLOUR_WINDOW_BACKGROUND)
            imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, Node.DEFAULT_NODE_BORDER_THICKNESS)
            imgui.push_style_color(imgui.COLOR_BORDER, *Node.COLOUR_WINDOW_BORDER)
            imgui.push_style_color(imgui.COLOR_BORDER_SHADOW, *Node.COLOUR_WINDOW_BORDER)
        imgui.push_style_color(imgui.COLOR_TEXT, *Node.COLOUR_TEXT)
        imgui.push_style_color(imgui.COLOR_HEADER, *self.colour_whiten(self.colour))
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *self.colour_whiten(self.colour))
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *self.colour_whiten(self.colour))
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *self.colour_whiten(self.colour))
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *Node.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_BUTTON, *self.colour)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *self.colour_brighten(self.colour))
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *self.colour_whiten(self.colour))
        imgui.push_style_color(imgui.COLOR_TAB_HOVERED, *self.colour_brighten(self.colour))
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, Node.WINDOW_ROUNDING)
        imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, Node.WINDOW_ROUNDING)
        imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, Node.FRAME_ROUNDING)
        imgui.push_style_var(imgui.STYLE_WINDOW_MIN_SIZE, (1, 1))
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *Node.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *Node.COLOUR_WINDOW_BACKGROUND)
        _, stay_open = imgui.begin(self.title + f"##{self.id}", True, imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_COLLAPSE)

        if imgui.is_window_focused() and not imgui.is_any_item_hovered():
            if cfg.set_active_node(self):
                self.on_gain_focus()
            self.position[0] = self.position[0] + cfg.node_move_requested[0]
            self.position[1] = self.position[1] + cfg.node_move_requested[1]
        self.position[0] += cfg.camera_move_requested[0]
        self.position[1] += cfg.camera_move_requested[1]
        self.last_measured_window_position = imgui.get_window_position()
        self.node_height = imgui.get_window_size()[1]
        imgui.push_clip_rect(0, 0, cfg.window_width, cfg.window_height)
        imgui.push_id(str(self.id))
        if not stay_open:
            self.render_end()
            self.delete()
            return False

        return True

    def render(self):
        if self.render_start():
            self.render_end()

    def render_end(self):
        ## Node context_menu:
        if imgui.begin_popup_context_window():
            self.keep_active = cfg.focused_node == self
            _changed, self.keep_active = imgui.checkbox("Focus node", self.keep_active)
            if _changed:
                if self.keep_active:
                    cfg.set_active_node(self, True)
                else:
                    cfg.set_active_node(None, True)
                    cfg.set_active_node(self, False)
                imgui.close_current_popup()
            # _duplicate, _ = imgui.menu_item("Duplicate node")
            # if _duplicate:
            #     duplicate_node = Node.create_node_by_type(self.type) ## TODO FIX
            #     duplicate_node.position = [self.position[0] + 10, self.position[1] + 10]
            #     cfg.set_active_node(duplicate_node)
            _delete, _ = imgui.menu_item("Delete node")
            if _delete:
                self.delete()
            _reset, _ = imgui.menu_item("Reset node")
            # if _reset:
            #     new_node = Node.create_node_by_type(self.type) ## TODO FIX
            #     new_node.position = self.position
            #     self.delete()
            if imgui.begin_menu("Set node-specific LUT"):
                for lut in ["auto"] + settings.lut_names:
                    _lut, _ = imgui.menu_item(lut)
                    if _lut:
                        self.lut = lut
                imgui.end_menu()
            imgui.end_popup()
        if cfg.profiling and self.does_profiling_time:
            imgui.separator()
            imgui.text(f"Time processing: {self.profiler_time:.2f}")
            imgui.text(f"Frames requested: {self.profiler_count}")
        imgui.pop_style_color(23)
        imgui.pop_style_var(5)
        imgui.pop_id()
        imgui.pop_clip_rect()
        imgui.end()

    def play_button(self):
        """
        Render a 'play / pause' button for the node, with play/pause state toggled by a single left mouse button click.
        :return: tuple (bool clicked, bool play). bool play is True is current state is 'play', False is 'pause'
        """
        clicked = False
        ## Start/stop button
        imgui.push_id(f"Startstop{self.id}")
        window_position = (self.last_measured_window_position[0] + self.size - Node.PLAY_BUTTON_SIZE / 2, self.last_measured_window_position[1] + self.node_height / 2 - Node.PLAY_BUTTON_SIZE / 2)

        imgui.set_next_window_position(*window_position)
        imgui.set_next_window_size(Node.PLAY_BUTTON_SIZE, Node.PLAY_BUTTON_SIZE)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *Node.PLAY_BUTTON_WINDOW_BACKGROUND)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 25.0)
        imgui.begin(f"##Node{self.id}playbtn", False,
                    imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_COLLAPSE)
        if imgui.is_window_hovered() and imgui.is_mouse_clicked(0, False):
            self.play = not self.play
            clicked = True
            self.any_change = True
        imgui.pop_style_color(1)
        imgui.pop_style_var(1)

        ## Draw play / pause symbols
        draw_list = imgui.get_window_draw_list()
        if self.play:
            # Draw STOP button
            origin = [window_position[0] + Node.PLAY_BUTTON_SIZE / 2.0,
                      window_position[1] + Node.PLAY_BUTTON_SIZE / 2.0]
            p1 = (-1.0 * Node.PLAY_BUTTON_ICON_SIZE, 1.5 * Node.PLAY_BUTTON_ICON_SIZE)
            p2 = (-1.0 * Node.PLAY_BUTTON_ICON_SIZE, -1.5 * Node.PLAY_BUTTON_ICON_SIZE)
            draw_list.add_polyline([(p1[0] + origin[0], p1[1] + origin[1]),
                                    (p2[0] + origin[0], p2[1] + origin[1])], imgui.get_color_u32_rgba(*Node.PLAY_BUTTON_ICON_COLOUR), False, Node.PLAY_BUTTON_ICON_LINEWIDTH / 2.0)
            p1 = (1.0 * Node.PLAY_BUTTON_ICON_SIZE, 1.5 * Node.PLAY_BUTTON_ICON_SIZE)
            p2 = (1.0 * Node.PLAY_BUTTON_ICON_SIZE, -1.5 * Node.PLAY_BUTTON_ICON_SIZE)
            draw_list.add_polyline([(p1[0] + origin[0], p1[1] + origin[1]),
                                    (p2[0] + origin[0], p2[1] + origin[1])],
                                   imgui.get_color_u32_rgba(*Node.PLAY_BUTTON_ICON_COLOUR), False,
                                   Node.PLAY_BUTTON_ICON_LINEWIDTH / 2.0)
        else:
            # Draw PLAY button
            origin = [window_position[0] + Node.PLAY_BUTTON_SIZE / 2.0,
                      window_position[1] + Node.PLAY_BUTTON_SIZE / 2.0]
            p1 = (Node.PLAY_BUTTON_ICON_SIZE * -0.65, Node.PLAY_BUTTON_ICON_SIZE)
            p2 = (Node.PLAY_BUTTON_ICON_SIZE * 0.85, 0)
            p3 = (Node.PLAY_BUTTON_ICON_SIZE * -0.65, -Node.PLAY_BUTTON_ICON_SIZE)
            draw_list.add_polyline([(p1[0] + origin[0], p1[1] + origin[1]),
                                    (p2[0] + origin[0], p2[1] + origin[1]),
                                    (p3[0] + origin[0], p3[1] + origin[1])], imgui.get_color_u32_rgba(*Node.PLAY_BUTTON_ICON_COLOUR), True, Node.PLAY_BUTTON_ICON_LINEWIDTH)
        imgui.end()
        imgui.pop_id()

        return clicked, self.play

    def progress_bar(self, progress):
        """
        Renders a progress bar at the current imgui cursor position.
        :param progress:float 0.0 to 1.0.
        :return:
        """
        width = imgui.get_content_region_available_width()
        origin = imgui.get_window_position()
        y = imgui.get_cursor_screen_pos()[1]
        drawlist = imgui.get_window_draw_list()
        drawlist.add_rect_filled(Node.PROGRESS_BAR_PADDING + origin[0], y,
                                 Node.PROGRESS_BAR_PADDING + origin[0] + width, y + Node.PROGRESS_BAR_HEIGHT,
                                 imgui.get_color_u32_rgba(*Node.colour_whiten(self.colour)))
        drawlist.add_rect_filled(Node.PROGRESS_BAR_PADDING + origin[0], y,
                                 Node.PROGRESS_BAR_PADDING + origin[0] + width * min([1.0, progress]),
                                 y + Node.PROGRESS_BAR_HEIGHT, imgui.get_color_u32_rgba(*self.colour))

    def delete(self):
        try:
            if cfg.focused_node == self:
                cfg.set_active_node(None, True)
            if cfg.active_node == self:
                cfg.set_active_node(None)
            if self in cfg.nodes:
                cfg.nodes.remove(self)
            for attribute in self.connectable_attributes:
                attribute.delete()
        except Exception as e:
            cfg.set_error(e, "Problem deleting node\n"+str(e))

    @staticmethod
    def tooltip(text):
        if imgui.is_item_hovered():
            if Node.TOOLTIP_HOVERED_TIMER == 0.0:
                Node.TOOLTIP_HOVERED_START_TIME = time.time()
                Node.TOOLTIP_HOVERED_TIMER = 0.001  # add a fake 1 ms to get out of this if clause
            elif Node.TOOLTIP_HOVERED_TIMER > Node.TOOLTIP_APPEAR_DELAY:
                imgui.set_tooltip(text)
            else:
                Node.TOOLTIP_HOVERED_TIMER = time.time() - Node.TOOLTIP_HOVERED_START_TIME
        if not imgui.is_any_item_hovered():
            Node.TOOLTIP_HOVERED_TIMER = 0.0

    @staticmethod
    def colour_brighten(c):
        return (c[0] * 1.2, c[1] * 1.2, c[2] * 1.2, 1.0)

    @staticmethod
    def colour_whiten(c):
        return (c[0] * 0.3 + 0.7, c[1] * 0.3 + 0.7, c[2] * 0.3 + 0.7, 1.0)

    @staticmethod
    def get_source_load_data_node(node):
        """
        This function is _static_ in order to ensure proper usage; it relies on a node having a connectable_attribute with type INPUT that ultimately is linked to a LoadDataNode.
        :param node: the node for which you want to find the source LoadDataNode.
        :return: node: node with type TYPE_LOAD_DATA
        """
        def get_any_incoming_node(_node):
            for attribute in _node.connectable_attributes:
                if attribute.direction == ConnectableAttribute.INPUT:
                    return attribute.get_incoming_node()

        if node.NODE_IS_DATA_SOURCE:
            return node
        try:
            source = get_any_incoming_node(node)
            while not isinstance(source, NullNode):
                if source.NODE_IS_DATA_SOURCE:
                    return source
                source = get_any_incoming_node(source)
            return source
        except Exception as e:
            return NullNode()

    def get_image(self, idx):
        """
        :param idx:int, index of frame in dataset
        :return: Frame object
        """
        retval = None
        if cfg.profiling:
            start_time = time.time()
            self.profiler_count += 1
        try:
            if self.buffer_last_output:
                if idx is self.last_index_requested:
                    retval = copy.deepcopy(self.last_frame_returned)
                else:
                    self.last_frame_returned = self.get_image_impl(idx)
                    self.last_index_requested = idx
                    retval = copy.deepcopy(self.last_frame_returned)
            retval = self.get_image_impl(idx)
        except Exception as e:
            cfg.set_error(e, f"{self} error: "+str(e))
        if cfg.profiling:
            self.profiler_time += (time.time() - start_time)
        return retval

    def get_image_impl(self, idx):
        return None

    def get_particle_data(self):
        return self.get_particle_data_impl()

    def get_particle_data_impl(self):
        return ParticleData()

    def get_coordinates(self, idx):
        return None

    def on_update(self):
        return None

    def on_gain_focus(self):
        self.NODE_GAINED_FOCUS = True

    def pre_save(self):
        cfg.pickle_temp = dict()
        self.last_index_requested = -1
        self.last_frame_returned = None
        cfg.pickle_temp["profiler_time"] = self.profiler_time
        self.pre_save_impl()

    def pre_save_impl(self):
        pass

    def post_save(self):
        self.profiler_time = cfg.pickle_temp["profiler_time"]
        self.post_save_impl()
        cfg.pickle_temp = dict()

    def post_save_impl(self):
        pass


class NullNode(Node):
    def __init__(self):
        super().__init__()
        self.dataset = Dataset()
        self.dataset_in = None
        self.pixel_size = 100
        cfg.nodes.remove(self)

    def __bool__(self):
        return False


class ConnectableAttribute:
    id_generator = count(0)
    active_connector = None

    TYPE_DATASET = 1
    TYPE_IMAGE = 2
    TYPE_RECONSTRUCTION = 3
    TYPE_COLOUR = 4
    TYPE_MULTI = 5
    TYPE_COORDINATES = 6

    INPUT = True
    OUTPUT = False

    COLOUR = dict()
    COLOUR[TYPE_DATASET] = (54 / 255, 47 / 255, 192 / 255, 1.0)
    COLOUR[TYPE_IMAGE] = (255 / 255, 179 / 255, 35 / 255, 1.0)
    COLOUR[TYPE_RECONSTRUCTION] = (243 / 255, 9 / 255, 9 / 255, 1.0)
    COLOUR[TYPE_COLOUR] = (4 / 255, 4 / 255, 4 / 255, 1.0)
    COLOUR[TYPE_MULTI] = (230 / 255, 230 / 255, 240 / 255, 1.0)
    COLOUR[TYPE_COORDINATES] = (230 / 255, 174 / 255, 13 / 255, 1.0)
    COLOUR_BORDER = (0.0, 0.0, 0.0, 1.0)

    TITLE = dict()
    TITLE[TYPE_DATASET] = "Dataset"
    TITLE[TYPE_IMAGE] = "Image"
    TITLE[TYPE_RECONSTRUCTION] = "Reconstruction"
    TITLE[TYPE_COLOUR] = "Colour"
    TITLE[TYPE_MULTI] = "Dataset"
    TITLE[TYPE_COORDINATES] = "Coordinates"

    CONNECTOR_SPACING = 10
    CONNECTOR_HORIZONTAL_OFFSET = -8
    CONNECTOR_VERTICAL_OFFSET = 7
    CONNECTOR_RADIUS = 5
    CONNECTOR_WINDOW_PADDING = 9
    CONNECTOR_SEGMENTS = 16
    CONNECTION_LINE_COLOUR = (0.0, 0.0, 0.0, 1.0)
    CONNECTION_LINE_THICKNESS = 2

    def __init__(self, attributetype, direction, parent, allowed_partner_types=None):
        self.id = int(datetime.datetime.utcnow().timestamp()) + next(ConnectableAttribute.id_generator)
        self.type = attributetype
        self.title = ConnectableAttribute.TITLE[self.type]
        self.colour = ConnectableAttribute.COLOUR[self.type]
        self.direction = direction
        self.linked_attributes = list()
        self.draw_y = 0
        self.draw_x = 0
        self.connector_position = [0, 0]
        self.newly_connected = False
        self.any_change = False
        self.newly_disconnected = False
        self.parent = parent
        self.multi = self.type == ConnectableAttribute.TYPE_MULTI
        self.allowed_partner_types = [self.type]
        self.current_type = self.type
        if allowed_partner_types is not None:
            self.allowed_partner_types = allowed_partner_types

        parent.connectable_attributes.append(self)

    def __eq__(self, other):
        if type(self) is type(other):
            return self.id == other.id
        return False

    def __str__(self):
        return f"Attribute type {self.type} with id {self.id}"

    def clear_flags(self):
        self.any_change = False

    def render_start(self):
        imgui.push_id(f"Attribute{self.id}")

    def render_end(self, show_label=True):
        text_width = imgui.get_font_size() * len(self.title) / 2
        window_width = imgui.get_window_content_region_width()
        self.draw_x = imgui.get_cursor_screen_pos()[0] + (ConnectableAttribute.CONNECTOR_HORIZONTAL_OFFSET if self.direction else window_width - 1 - ConnectableAttribute.CONNECTOR_HORIZONTAL_OFFSET)
        if self.direction == ConnectableAttribute.INPUT:
            imgui.same_line(position = ConnectableAttribute.CONNECTOR_SPACING)
            self.draw_y = imgui.get_cursor_screen_pos()[1] + ConnectableAttribute.CONNECTOR_VERTICAL_OFFSET
        else:
            imgui.same_line(position=window_width - text_width)
            self.draw_y = imgui.get_cursor_screen_pos()[1] + ConnectableAttribute.CONNECTOR_VERTICAL_OFFSET

        if show_label:
            imgui.text(self.title)
        imgui.pop_id()

        ## connector drawing
        self.connector_position = (self.draw_x, self.draw_y)
        draw_list = imgui.get_window_draw_list()
        draw_list.add_circle_filled(self.connector_position[0], self.connector_position[1],
                                    ConnectableAttribute.CONNECTOR_RADIUS + 1, imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 1.0),
                                    ConnectableAttribute.CONNECTOR_SEGMENTS)
        draw_list.add_circle_filled(self.connector_position[0], self.connector_position[1], ConnectableAttribute.CONNECTOR_RADIUS, imgui.get_color_u32_rgba(*self.colour), ConnectableAttribute.CONNECTOR_SEGMENTS)

        ## Connector user input
        self.connector_drag_drop_sources()

        if self.direction == ConnectableAttribute.OUTPUT:
            for partner in self.linked_attributes:
                draw_list = imgui.get_background_draw_list()
                draw_list.add_line(self.draw_x, self.draw_y, partner.draw_x, partner.draw_y, imgui.get_color_u32_rgba(*ConnectableAttribute.CONNECTION_LINE_COLOUR), ConnectableAttribute.CONNECTION_LINE_THICKNESS)
        if cfg.active_connector == self:
            if cfg.active_connector_hover_pos is not None:
                draw_list = imgui.get_background_draw_list()
                draw_list.add_line(self.draw_x, self.draw_y, *cfg.active_connector_hover_pos, imgui.get_color_u32_rgba(*ConnectableAttribute.CONNECTION_LINE_COLOUR), ConnectableAttribute.CONNECTION_LINE_THICKNESS)
        if self.any_change:
            self.parent.any_change = True

    def get_incoming_node(self):
        if self.direction == ConnectableAttribute.INPUT:
            if len(self.linked_attributes) == 1:
                return self.linked_attributes[0].parent
        return NullNode()

    def get_incoming_attribute_type(self):
        if self.direction == ConnectableAttribute.INPUT:
            if len(self.linked_attributes) == 1:
                return self.linked_attributes[0].type
        return None

    def connector_drag_drop_sources(self):
        ## 'render' the invisible node drag/drop sources]
        connector_window_size = (ConnectableAttribute.CONNECTOR_RADIUS + ConnectableAttribute.CONNECTOR_WINDOW_PADDING) * 2 + 1
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_color(imgui.COLOR_BORDER, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_color(imgui.COLOR_BORDER_SHADOW, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_color(imgui.COLOR_DRAG_DROP_TARGET, 0.0, 0.0, 0.0, 1.0)

        imgui.set_next_window_size(connector_window_size, connector_window_size)
        imgui.set_next_window_position(self.draw_x - connector_window_size // 2,
                                       self.draw_y - connector_window_size // 2)
        imgui.begin(f"Attribute{self.id}", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SAVED_SETTINGS)

        if imgui.is_window_hovered():
            if cfg.connector_delete_requested:
                cfg.connector_delete_requested = False
                self.disconnect_all()
                self.any_change = True

        ## Allow draw source if currently no node being edited
        if cfg.active_connector is self or cfg.active_connector is None:
            if imgui.begin_drag_drop_source(imgui.DRAG_DROP_SOURCE_NO_PREVIEW_TOOLTIP):
                cfg.active_connector = self
                cfg.active_connector_parent_node = self.parent
                imgui.set_drag_drop_payload('connector', b'1')  # Arbitrary payload value - manually keeping track of the payload elsewhere
                imgui.end_drag_drop_source()
        elif cfg.active_connector is not None:
            imgui.begin_child(f"Attribute{self.id}drop_target")
            imgui.end_child()
            if imgui.begin_drag_drop_target():
                if cfg.connector_released:
                    self.connect_attributes(cfg.active_connector)
                    cfg.connector_released = False
                imgui.end_drag_drop_target()
        imgui.end()
        imgui.pop_style_color(4)

    def connect_attributes(self, partner):
        io_match = self.direction != partner.direction
        type_match = self.type == partner.type or partner.type in self.allowed_partner_types or self.type in partner.allowed_partner_types
        novel_match = partner not in self.linked_attributes
        parent_match = self.parent != partner.parent
        if io_match and type_match and novel_match and parent_match:
            self.any_change = True
            self.parent.any_change = True
            partner.any_change = True
            partner.parent.any_change = True
            self.linked_attributes.append(partner)
            partner.linked_attributes.append(self)
            self.on_connect()
            partner.on_connect()
        cfg.active_connector_parent_node = None

    def disconnect_all(self):
        for partner in self.linked_attributes:
            partner.linked_attributes.remove(self)
            partner.notify_disconnect()
        self.notify_disconnect()
        self.linked_attributes = list()
        self.parent.any_change = True
        self.colour = ConnectableAttribute.COLOUR[self.type]
        self.title = ConnectableAttribute.TITLE[self.type]

    def is_connected(self):
        return len(self.linked_attributes) > 1

    def delete(self):
        self.disconnect_all()

    def on_connect(self):
        ## when on connect is called, a new ConnectableAttribute has just been added to the linked_partners list
        if self.direction == ConnectableAttribute.INPUT:
            for idx in range(len(self.linked_attributes) - 1):
                self.linked_attributes[idx].notify_disconnect()
                self.linked_attributes[idx].linked_attributes.remove(self)
                self.linked_attributes.remove(self.linked_attributes[idx])
        self.newly_connected = True
        self.parent.any_change = True
        if self.type == ConnectableAttribute.TYPE_MULTI:
            self.colour = self.linked_attributes[-1].colour
            self.current_type = self.linked_attributes[-1].type
            self.title = self.linked_attributes[-1].title

    def notify_disconnect(self):
        self.any_change = True
        self.newly_disconnected = True
        self.colour = ConnectableAttribute.COLOUR[self.type]
        self.title = ConnectableAttribute.TITLE[self.type]

    def check_connect_event(self):
        if self.newly_connected:
            self.newly_connected = False
            return True
        return False

    def check_disconnect_event(self):
        if self.newly_disconnected:
            self.newly_disconnected = False
            return True
        return False


