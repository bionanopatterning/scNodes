from itertools import count
import imgui
from imgui.integrations.glfw import GlfwRenderer
import config as cfg
import glfw
from opengl_classes import *
import tkinter as tk
from tkinter import filedialog
from dataset import *
from reconstruction import *
from util import *
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.signal import medfilt
from skimage.feature import peak_local_max
from pystackreg import StackReg
import pywt
from joblib import Parallel, delayed, cpu_count
import particlefitting as pfit
tkroot = tk.Tk()
tkroot.withdraw()


class NodeEditor:
    # TODO: pickling node setups - for debug as well as for final use
    if True:
        COLOUR_WINDOW_BACKGROUND = (0.94, 0.94, 0.94, 1.0)
        CONTEXT_MENU_SIZE = (200, 100)
        COLOUR_ERROR_WINDOW_BACKGROUND = (0.84, 0.84, 0.84, 1.0)
        COLOUR_ERROR_WINDOW_HEADER = (0.6, 0.3, 0.3, 1.0)
        COLOUR_ERROR_WINDOW_HEADER_NEW = (0.8, 0.35, 0.35, 1.0)
        COLOUR_ERROR_WINDOW_TEXT = (0.0, 0.0, 0.0, 1.0)
        COLOUR_CM_WINDOW_BACKGROUND = (0.96, 0.96, 0.96, 1.0)
        COLOUR_CM_WINDOW_TEXT = (0.0, 0.0, 0.0, 1.0)
        COLOUR_CM_OPTION_HOVERED = (1.0, 1.0, 1.0, 1.0)

    nodes = list()
    active_node = None
    next_active_node = None
    connectable_attributes = list()
    active_connector = None
    active_connector_hover_pos = [0, 0]
    connector_released = False
    connector_delete_requested = False
    node_move_requested = [0, 0]
    camera_move_requested = [0, 0]

    error_msg = None
    error_new = True
    error_obj = None


    def __init__(self, window, shared_font_atlas=None):
        self.window = window
        self.window.clear_color = NodeEditor.COLOUR_WINDOW_BACKGROUND
        self.window.make_current()

        if shared_font_atlas is not None:
            self.imgui_context = imgui.create_context(shared_font_atlas)
        else:
            imgui.get_current_context()
            self.imgui_context = imgui.create_context()
        self.imgui_implementation = GlfwRenderer(self.window.glfw_window)

        self.window.set_mouse_callbacks()

        # Context menu
        self.context_menu_position = [0, 0]
        self.context_menu_open = False
        self.context_menu_can_close = False


    def get_font_atlas_ptr(self):
        return self.imgui_implementation.io.fonts

    def on_update(self):
        if NodeEditor.next_active_node is not None:
            NodeEditor.active_node = NodeEditor.next_active_node
            NodeEditor.next_active_node = None

        imgui.set_current_context(self.imgui_context)
        self.window.make_current()
        self.window.set_full_viewport()
        if self.window.focused:
            self.imgui_implementation.process_inputs()
        self.window.on_update()

        for node in NodeEditor.nodes:
            node.on_update()

        NodeEditor.connector_delete_requested = False
        NodeEditor.active_connector_hover_pos = self.window.cursor_pos
        if NodeEditor.active_connector is not None:
            if not self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT):
                NodeEditor.connector_released = True
                NodeEditor.active_connector_hover_pos = None
                NodeEditor.active_connector = None
            else:
                NodeEditor.connector_released = False
        if self.window.get_mouse_event(glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, pop_event = False):
            NodeEditor.connector_delete_requested = True

        NodeEditor.node_move_requested = [0, 0]
        NodeEditor.camera_move_requested = [0, 0]
        if self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT):
            if imgui.get_io().want_capture_mouse:
                NodeEditor.node_move_requested = self.window.cursor_delta
            else:
                NodeEditor.camera_move_requested = self.window.cursor_delta




        imgui.new_frame()
        self._gui_main()
        imgui.render()
        self.imgui_implementation.render(imgui.get_draw_data())

    def end_frame(self):
        self.window.end_frame()

    def _gui_main(self):

        ## Render nodes
        if NodeEditor.active_node is not None:
            NodeEditor.active_node.render()
        for node in NodeEditor.nodes:
            if node is not NodeEditor.active_node:
                node.render()

        ## Context menu
        if not self.context_menu_open:
            if self.window.get_mouse_event(glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0) and not imgui.get_io().want_capture_mouse:
                self.context_menu_position = self.window.cursor_pos
                self.context_menu_open = True
                self.context_menu_can_close = False
        else:
            self._context_menu()

        ## Error message
        if NodeEditor.error_msg is not None:
            if NodeEditor.error_new:
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *NodeEditor.COLOUR_ERROR_WINDOW_HEADER_NEW)
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *NodeEditor.COLOUR_ERROR_WINDOW_HEADER_NEW)
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *NodeEditor.COLOUR_ERROR_WINDOW_HEADER_NEW)
            else:
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *NodeEditor.COLOUR_ERROR_WINDOW_HEADER)
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *NodeEditor.COLOUR_ERROR_WINDOW_HEADER)
                imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *NodeEditor.COLOUR_ERROR_WINDOW_HEADER)
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *NodeEditor.COLOUR_ERROR_WINDOW_BACKGROUND)
            imgui.push_style_color(imgui.COLOR_TEXT, *NodeEditor.COLOUR_ERROR_WINDOW_TEXT)
            imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 3.0)
            _, stay_open = imgui.begin("Error", True)
            imgui.text("Action failed with error:")
            imgui.text(NodeEditor.error_msg)
            if imgui.button("(debug): raise error", 180, 20):
                raise(NodeEditor.error_obj)
            if imgui.is_window_focused() and self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.PRESS):
               NodeEditor.error_new = False
            imgui.end()
            if not stay_open:
                NodeEditor.error_msg = None
                NodeEditor.error_new = True
            imgui.pop_style_color(5)
            imgui.pop_style_var(1)

    def _context_menu(self):

        ## Open/close logic & start window
        imgui.set_next_window_position(self.context_menu_position[0] - 3, self.context_menu_position[1] - 3)
        imgui.set_next_window_size(*NodeEditor.CONTEXT_MENU_SIZE)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *NodeEditor.COLOUR_CM_WINDOW_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_TEXT, *NodeEditor.COLOUR_CM_WINDOW_TEXT)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *NodeEditor.COLOUR_CM_WINDOW_BACKGROUND)
        imgui.begin("##necontextmenu", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE)
        # Close context menu when it is not hovered.
        context_menu_hovered = imgui.is_window_hovered(flags=imgui.HOVERED_ALLOW_WHEN_BLOCKED_BY_POPUP | imgui.HOVERED_CHILD_WINDOWS)
        if context_menu_hovered:
            self.context_menu_can_close = True
        if not context_menu_hovered and self.context_menu_can_close:
            self.context_menu_can_close = False
            self.context_menu_open = False

        # Context menu contents
        new_node = None
        if imgui.begin_menu("Data I/O"):
            for key in [Node.TYPE_LOAD_DATA, Node.TYPE_EXPORT_DATA]:
                item_selected, _ = imgui.menu_item(Node.TITLE[key])
                if item_selected:
                    new_node = self.create_node_by_type(key)
            imgui.end_menu()
        if imgui.begin_menu("Image processing"):
            for key in [Node.TYPE_REGISTER, Node.TYPE_SPATIAL_FILTER, Node.TYPE_TEMPORAL_FILTER, Node.TYPE_FRAME_SELECTION, Node.TYPE_FRAME_SHIFT, Node.TYPE_BIN_IMAGE, Node.TYPE_IMAGE_CALCULATOR, Node.TYPE_GET_IMAGE]:
                item_selected, _ = imgui.menu_item(Node.TITLE[key])
                if item_selected:
                    new_node = self.create_node_by_type(key)
            imgui.end_menu()
        if imgui.begin_menu("Reconstruction nodes"):
            for key in [Node.TYPE_PARTICLE_DETECTION, Node.TYPE_PARTICLE_FITTING, Node.TYPE_PARTICLE_FILTER, Node.TYPE_PARTICLE_PAINTER, Node.TYPE_RECONSTRUCTOR]:
                item_selected, _ = imgui.menu_item(Node.TITLE[key])
                if item_selected:
                    new_node = self.create_node_by_type(key)
            imgui.end_menu()

        if new_node is not None:
            try:
                new_node.position = self.context_menu_position
                NodeEditor.active_node = new_node
                self.context_menu_open = False
            except Exception as e:
                NodeEditor.set_error(e, "Error upon requesting new node (probably not implemented yet.)\nError:"+str(e))


        # End
        imgui.pop_style_color(3)
        imgui.end()

    @staticmethod
    def set_active_node(node):
        if node is None:
            NodeEditor.active_node = None
        else:
            NodeEditor.next_active_node = node

    @staticmethod
    def create_node_by_type(node_type):
        if node_type == Node.TYPE_LOAD_DATA:
            return LoadDataNode()
        elif node_type == Node.TYPE_REGISTER:
            return RegisterNode()
        elif node_type == Node.TYPE_SPATIAL_FILTER:
            return SpatialFilterNode()
        elif node_type == Node.TYPE_TEMPORAL_FILTER:
            return TemporalFilterNode()
        elif node_type == Node.TYPE_FRAME_SELECTION:
            return FrameSelectionNode()
        elif node_type == Node.TYPE_RECONSTRUCTOR:
            return ReconstructionRendererNode()
        elif node_type == Node.TYPE_PARTICLE_PAINTER:
            return ParticlePainterNode()
        elif node_type == Node.TYPE_GET_IMAGE:
            return GetImageNode()
        elif node_type == Node.TYPE_IMAGE_CALCULATOR:
            return ImageCalculatorNode()
        elif node_type == Node.TYPE_PARTICLE_DETECTION:
            return ParticleDetectionNode()
        elif node_type == Node.TYPE_EXPORT_DATA:
            return ExportDataNode()
        elif node_type == Node.TYPE_FRAME_SHIFT:
            return FrameShiftNode()
        elif node_type == Node.TYPE_BIN_IMAGE:
            return BinImageNode()
        elif node_type == Node.TYPE_PARTICLE_FITTING:
            return ParticleFittingNode()
        else:
            return False

    @staticmethod
    def set_error(error_obj, error_msg):
        NodeEditor.error_msg = error_msg
        NodeEditor.error_obj = error_obj
        NodeEditor.error_new = True


class Node:
    if True:
        id_generator = count(0)

        TYPE_NULL = -1
        TYPE_NONE = 0
        TYPE_LOAD_DATA = 1
        TYPE_REGISTER = 2
        TYPE_SPATIAL_FILTER = 3
        TYPE_TEMPORAL_FILTER = 4
        TYPE_FRAME_SELECTION = 5
        TYPE_PARTICLE_DETECTION = 6
        TYPE_PARTICLE_FITTING = 7
        TYPE_PARTICLE_FILTER = 8
        TYPE_RECONSTRUCTOR = 9
        TYPE_PARTICLE_PAINTER = 10
        TYPE_EXPORT_DATA = 11
        TYPE_GET_IMAGE = 12
        TYPE_IMAGE_CALCULATOR = 13
        TYPE_FRAME_SHIFT = 14
        TYPE_BIN_IMAGE = 15

        COLOUR = dict()
        COLOUR[TYPE_LOAD_DATA] = (54 / 255, 47 / 255, 192 / 255, 1.0)
        COLOUR[TYPE_REGISTER] = (68 / 255, 177 / 255, 209 / 255, 1.0)
        COLOUR[TYPE_SPATIAL_FILTER] = (44 / 255, 217 / 255, 158 / 255, 1.0)
        COLOUR[TYPE_TEMPORAL_FILTER] = (55 / 255, 236 / 255, 54 / 255, 1.0)
        COLOUR[TYPE_FRAME_SELECTION] = (228 / 255, 231 / 255, 59 / 255, 1.0)
        COLOUR[TYPE_PARTICLE_DETECTION] = (230 / 255, 174 / 255, 13 / 255, 1.0)
        COLOUR[TYPE_PARTICLE_FITTING] = (230 / 255, 98 / 255, 13 / 255, 1.0)
        COLOUR[TYPE_PARTICLE_FILTER] = (230 / 255, 13 / 255, 13 / 255, 1.0)
        COLOUR[TYPE_RECONSTRUCTOR] = (243 / 255, 0 / 255, 80 / 255, 1.0)
        COLOUR[TYPE_PARTICLE_PAINTER] = (143 / 255, 143 / 255, 143 / 255, 1.0)
        COLOUR[TYPE_EXPORT_DATA] = (138 / 255, 8 / 255, 8 / 255, 1.0)
        COLOUR[TYPE_GET_IMAGE] = (143 / 255, 143 / 255, 143 / 255, 1.0)
        COLOUR[TYPE_IMAGE_CALCULATOR] = (143 / 255, 123 / 255, 103 / 255, 1.0)
        COLOUR[TYPE_FRAME_SHIFT] = (50 / 255, 223 / 255, 80 / 255, 1.0)
        COLOUR[TYPE_BIN_IMAGE] = (143 / 255, 123 / 255, 103 / 255, 1.0)
        COLOUR[TYPE_NULL] = (1.0, 0.0, 1.0, 1.0)

        COLOUR_WINDOW_BACKGROUND = (0.96, 0.96, 0.96, 0.96)
        COLOUR_WINDOW_BORDER = (0.45, 0.45, 0.45, 1.0)
        COLOUR_WINDOW_BORDER_ACTIVE_NODE = (0.0, 0.0, 0.0, 1.0)

        COLOUR_TEXT = (0.0, 0.0, 0.0, 1.0)
        COLOUR_FRAME_BACKGROUND = (0.84, 0.84, 0.84, 1.0)


        TITLE = dict()
        TITLE[TYPE_LOAD_DATA] = "Load data"
        TITLE[TYPE_REGISTER] = "Registration"
        TITLE[TYPE_SPATIAL_FILTER] = "Spatial filter"
        TITLE[TYPE_TEMPORAL_FILTER] = "Temporal filter"
        TITLE[TYPE_FRAME_SELECTION] = "Frame filter"
        TITLE[TYPE_PARTICLE_DETECTION] = "Particle detection"
        TITLE[TYPE_PARTICLE_FITTING] = "Particle fitting"
        TITLE[TYPE_PARTICLE_FILTER] = "Particle filter"
        TITLE[TYPE_RECONSTRUCTOR] = "Reconstruction renderer"
        TITLE[TYPE_PARTICLE_PAINTER] = "Particle painter"
        TITLE[TYPE_EXPORT_DATA] = "Export data"
        TITLE[TYPE_GET_IMAGE] = "Dataset to image"
        TITLE[TYPE_IMAGE_CALCULATOR] = "Image calculator"
        TITLE[TYPE_FRAME_SHIFT] = "Frame shift"
        TITLE[TYPE_BIN_IMAGE] = "Bin image"
        TITLE[TYPE_NULL] = "null"
        WINDOW_ROUNDING = 5.0

        PLAY_BUTTON_SIZE = 40
        PLAY_BUTTON_ICON_SIZE = 5.0
        PLAY_BUTTON_ICON_LINEWIDTH = 12.0
        PLAY_BUTTON_ICON_COLOUR = (0.2, 0.8, 0.2, 1.0)

        PROGRESS_BAR_HEIGHT = 10
        PROGRESS_BAR_PADDING = 8

    def __init__(self, nodetype):
        self.id = next(Node.id_generator)
        self.type = nodetype
        self.position = [0, 0]
        self.last_measured_window_position = [0, 0]
        self.size = [0, 0]
        self.connectable_attributes = list()
        self.colour = Node.COLOUR[self.type]
        self.title = Node.TITLE[self.type]
        self.play = True
        self.any_change = False
        self.queued_actions = list()
        self.use_roi = False
        if self.type is not Node.TYPE_NULL:
            NodeEditor.nodes.append(self)

        # bookkeeping
        self.last_index_requested = -1
        self.last_frame_returned = None

    def __eq__(self, other):
        if type(self) is type(other):
            return self.id == other.id
        return False

    def render_start(self):
        self.any_change = False
        ## render the node window
        imgui.set_next_window_size(*self.size, imgui.ONCE)
        imgui.set_next_window_position(self.position[0], self.position[1], imgui.ALWAYS)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *Node.COLOUR[self.type])
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *Node.COLOUR[self.type])
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *Node.COLOUR[self.type])
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *Node.COLOUR[self.type])
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *Node.COLOUR_WINDOW_BACKGROUND)
        if NodeEditor.active_node == self:
            imgui.push_style_color(imgui.COLOR_BORDER, *Node.COLOUR_WINDOW_BORDER_ACTIVE_NODE)
            imgui.push_style_color(imgui.COLOR_BORDER_SHADOW, *Node.COLOUR_WINDOW_BORDER_ACTIVE_NODE)
        else:
            imgui.push_style_color(imgui.COLOR_BORDER, *Node.COLOUR_WINDOW_BORDER)
            imgui.push_style_color(imgui.COLOR_BORDER_SHADOW, *Node.COLOUR_WINDOW_BORDER)
        imgui.push_style_color(imgui.COLOR_TEXT, *Node.COLOUR_TEXT)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *self.colour_whiten(Node.COLOUR[self.type]))
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *Node.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *Node.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_BUTTON, *Node.COLOUR[self.type])
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *self.colour_brighten(Node.COLOUR[self.type]))
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *self.colour_whiten(Node.COLOUR[self.type]))
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *Node.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, Node.WINDOW_ROUNDING)
        imgui.push_style_var(imgui.STYLE_WINDOW_MIN_SIZE, (1, 1))
        _, stay_open = imgui.begin(self.title + f"##{self.id}", True, imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_MOVE)
        if imgui.is_window_focused() and not imgui.is_any_item_hovered():
            NodeEditor.set_active_node(self)
            self.position[0] = self.position[0] + NodeEditor.node_move_requested[0]
            self.position[1] = self.position[1] + NodeEditor.node_move_requested[1]
        self.position[0] += NodeEditor.camera_move_requested[0]
        self.position[1] += NodeEditor.camera_move_requested[1]
        self.last_measured_window_position = imgui.get_window_position()
        imgui.push_clip_rect(0, 0, 1920, 1080)
        imgui.push_id(str(self.id))

        if not stay_open:
            self.render_end()
            self.delete()
            return False

        return True

    def render(self):
        pass # to be implemented per Node type.

    def render_end(self):
        imgui.pop_style_color(15)
        imgui.pop_style_var(2)
        imgui.pop_id()
        imgui.pop_clip_rect()
        imgui.end()

    def play_button(self):
        ## Start/stop button
        any_change = False
        imgui.push_id(f"Startstop{self.id}")
        window_position = (self.last_measured_window_position[0] + self.size[0] - Node.PLAY_BUTTON_SIZE / 2, self.last_measured_window_position[1] + self.size[1] / 2 - Node.PLAY_BUTTON_SIZE / 2)

        imgui.set_next_window_position(*window_position)
        imgui.set_next_window_size(Node.PLAY_BUTTON_SIZE, Node.PLAY_BUTTON_SIZE)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *Node.COLOUR_WINDOW_BACKGROUND)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 25.0)
        imgui.begin(f"##Node{self.id}playbtn", False,
                    imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_COLLAPSE)
        if imgui.is_window_hovered() and imgui.is_mouse_clicked(0, False):
            self.play = not self.play
            any_change = True
        imgui.pop_style_color(1)
        imgui.pop_style_var(1)

        ## Draw play / pause symbols

        draw_list = imgui.get_window_draw_list()
        if self.play:
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
            # play button
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

        return any_change

    def delete(self):

        if NodeEditor.active_node == self:
            NodeEditor.set_active_node(None)
        NodeEditor.nodes.remove(self)
        for attribute in self.connectable_attributes:
            attribute.delete()

    @staticmethod
    def colour_brighten(c):
        return (c[0] * 1.2, c[1] * 1.2, c[2] * 1.2, 1.0)

    @staticmethod
    def colour_whiten(c):
        return (c[0] * 1.8, c[1] * 1.8, c[2] * 1.8, 1.0)

    @staticmethod
    def get_source_load_data_node(node):
        """
        This function is _static_ in order to ensure proper usage; it relies on a node having a dataset_in member variable of type ConnectableAttribute, which is ultimately linked to some LoadDataNode().
        :param node: the node for which you want to find the source LoadDataNode.
        :return: node: node with TYPE_LOAD_DATA
        """
        if node.type == Node.TYPE_LOAD_DATA:
            return node
        try:
            source = node.dataset_in.get_incoming_node()
            while not isinstance(source, NullNode):
                load_data_node = source.type == Node.TYPE_LOAD_DATA
                if load_data_node:
                    return source
                source = source.dataset_in.get_incoming_node()
            return source
        except Exception as e:
            NodeEditor.set_error(e, "While searching through connection tree for source LoadDataNode, the following\nerror was encountered:\n"+str(e))
            return NullNode()

    # 'Pure virtual' classes below - not sure how to actually make them virtual.
    def get_image(self, idx):
        if self.type == Node.TYPE_LOAD_DATA:
            return self.get_image_impl(idx)
        else:
            if idx is self.last_index_requested:
                return self.last_frame_returned
            else:
                try:
                    outframe = self.get_image_impl(idx)
                    self.last_frame_returned = outframe
                    self.last_index_requested = idx
                    return outframe
                except Exception as e:
                    NodeEditor.set_error(e, f"{Node.TITLE[self.type]} error: "+str(e))

    def get_image_impl(self, idx):
        return None

    def on_update(self):
        return None


class NullNode(Node):
    def __init__(self):
        super().__init__(Node.TYPE_NULL)
        self.dataset = Dataset()
        self.dataset_in = None




class ConnectableAttribute:
    id_generator = count(0)
    active_connector = None

    TYPE_DATASET = 1
    TYPE_IMAGE = 2
    TYPE_RECONSTRUCTION = 3
    TYPE_COLOUR = 4
    TYPE_MULTI = 5

    INPUT = True
    OUTPUT = False

    COLOUR = dict()
    COLOUR[TYPE_DATASET] = (54 / 255, 47 / 255, 192 / 255, 1.0)
    COLOUR[TYPE_IMAGE] = (255 / 255, 179 / 255, 35 / 255, 1.0)
    COLOUR[TYPE_RECONSTRUCTION] = (243 / 255, 9 / 255, 9 / 255, 1.0)
    COLOUR[TYPE_COLOUR] = (7 / 255, 202 / 255, 16 / 255, 1.0)
    COLOUR[TYPE_MULTI] = (54 / 255, 47 / 255, 192 / 255, 1.0)

    COLOUR_BORDER = (0.0, 0.0, 0.0, 1.0)

    TITLE = dict()
    TITLE[TYPE_DATASET] = "Dataset"
    TITLE[TYPE_IMAGE] = "Image"
    TITLE[TYPE_RECONSTRUCTION] = "Reconstruction"
    TITLE[TYPE_COLOUR] = "Colour"
    TITLE[TYPE_MULTI] = "Dataset"

    CONNECTOR_SPACING = 10
    CONNECTOR_HORIZONTAL_OFFSET = -8
    CONNECTOR_VERTICAL_OFFSET = 7
    CONNECTOR_RADIUS = 5
    CONNECTOR_WINDOW_PADDING = 9 # CONNECTOR_RADIUS + CONNECTOR_WINDOW_PADDING must be odd for ideal alignment of connector dot & drag/drop imgui window
    CONNECTOR_SEGMENTS = 16
    CONNECTION_LINE_COLOUR = (0.0, 0.0, 0.0, 1.0)
    CONNECTION_LINE_THICKNESS = 2

    def __init__(self, type, direction, parent, allowed_partner_types=None):
        self.id = next(ConnectableAttribute.id_generator)
        self.type = type
        self.title = ConnectableAttribute.TITLE[self.type]
        self.colour = ConnectableAttribute.COLOUR[self.type]
        self.direction = direction
        self.linked_attributes = list()
        self.draw_y = 0
        self.draw_x = 0
        self.connector_position = [0, 0]
        self.newly_connected = False
        self.any_change = False
        self.parent = parent
        self.multi = self.type == ConnectableAttribute.TYPE_MULTI
        self.allowed_partner_types = [self.type]
        self.current_type = self.type
        if allowed_partner_types is not None:
            self.allowed_partner_types = allowed_partner_types
        NodeEditor.connectable_attributes.append(self)

    def __eq__(self, other):
        if type(self) is type(other):
            return self.id == other.id
        return False

    def __str__(self):
        return f"Attribute type {self.type} with id {self.id}"

    def render_start(self):
        self.newly_connected = False
        self.any_change = False
        imgui.push_id(f"Attribute{self.id}")

    def render_end(self, show_label=True):
        any_change = False
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
        any_change = any_change | self.connector_logic()
        if self.direction == ConnectableAttribute.OUTPUT:
            for partner in self.linked_attributes:
                draw_list = imgui.get_background_draw_list()
                draw_list.add_line(self.draw_x, self.draw_y, partner.draw_x, partner.draw_y, imgui.get_color_u32_rgba(*ConnectableAttribute.CONNECTION_LINE_COLOUR), ConnectableAttribute.CONNECTION_LINE_THICKNESS)
        if NodeEditor.active_connector == self:
            if NodeEditor.active_connector_hover_pos is not None:
                draw_list = imgui.get_background_draw_list()
                draw_list.add_line(self.draw_x, self.draw_y, *NodeEditor.active_connector_hover_pos, imgui.get_color_u32_rgba(*ConnectableAttribute.CONNECTION_LINE_COLOUR), ConnectableAttribute.CONNECTION_LINE_THICKNESS)
        if any_change:
            self.parent.any_change = True
        return any_change

    def get_incoming_node(self):
        if self.direction == ConnectableAttribute.INPUT:
            if len(self.linked_attributes) == 1:
                return self.linked_attributes[0].parent
        return NullNode()

    def connector_logic(self):
        any_change = False
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
        ## TODO: delete all connections with a middle mouse click
        if imgui.is_window_hovered():
            if NodeEditor.connector_delete_requested:
                NodeEditor.connector_delete_requested = False
                self.disconnect_all()
                any_change = True

        ## Allow draw source if currently no node being edited
        if NodeEditor.active_connector is self or NodeEditor.active_connector is None:
            if imgui.begin_drag_drop_source(imgui.DRAG_DROP_SOURCE_NO_PREVIEW_TOOLTIP):
                NodeEditor.active_connector = self
                NodeEditor.set_active_node(self.parent)
                imgui.set_drag_drop_payload('connector', b'1')  # Arbitrary payload value - manually keeping track of the payload elsewhere
                imgui.end_drag_drop_source()
        elif not NodeEditor.active_connector is None:
            imgui.begin_child(f"Attribute{self.id}drop_target")
            imgui.end_child()
            if imgui.begin_drag_drop_target():
                if NodeEditor.connector_released:
                    any_change = any_change | self.connect_attributes(NodeEditor.active_connector)
                    NodeEditor.connector_released = False
                imgui.end_drag_drop_target()
        imgui.end()
        imgui.pop_style_color(4)
        return any_change

    def connect_attributes(self, partner):
        any_change = False
        io_match = self.direction != partner.direction
        type_match = self.type == partner.type or partner.type in self.allowed_partner_types or self.type in partner.allowed_partner_types
        novel_match = not partner in self.linked_attributes
        parent_match = self.parent != partner.parent
        if io_match and type_match and novel_match and parent_match:
            any_change = True
            self.parent.any_change = self.parent.any_change | True
            if self.direction == ConnectableAttribute.INPUT:
                self.disconnect_all()
            else:
                partner.disconnect_all()
            self.linked_attributes.append(partner)
            partner.linked_attributes.append(self)
            self.on_connect()
            partner.on_connect()
        NodeEditor.active_connector = None
        return any_change

    def disconnect_all(self):
        for partner in self.linked_attributes:
            partner.linked_attributes.remove(self)
            partner.notify_disconnect()
        self.linked_attributes = list()
        self.parent.any_change = True
        self.colour = ConnectableAttribute.COLOUR[self.type]
        self.title = ConnectableAttribute.TITLE[self.type]


    def is_connected(self):
        return len(self.linked_attributes) > 1

    def delete(self):
        self.disconnect_all()
        NodeEditor.connectable_attributes.remove(self)

    def on_connect(self):
        self.newly_connected = True
        self.parent.any_change = True
        if self.type == ConnectableAttribute.TYPE_MULTI:
            self.colour = self.linked_attributes[-1].colour
            self.current_type = self.linked_attributes[-1].type
            self.title = self.linked_attributes[-1].title

    def notify_disconnect(self):
        self.any_change = True
        self.colour = ConnectableAttribute.COLOUR[self.type]
        self.title = ConnectableAttribute.TITLE[self.type]


class LoadDataNode(Node):

    def __init__(self):
        super().__init__(Node.TYPE_LOAD_DATA) #Was: super(LoadDataNode, self).__init__()
        self.size = [200, 200]

        # Set up connectable attributes
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.connectable_attributes.append(self.dataset_out)
        # Set up node-specific vars
        self.dataset = Dataset()
        self.path = ""
        self.pixel_size = 64.0
        self.load_on_the_fly = True
        self.done_loading = False
        self.to_load_idx = 0
        self.n_to_load = 1

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_out.render_end()
            imgui.separator()
            imgui.text("Select source file")
            imgui.push_item_width(150)
            _, self.path = imgui.input_text("##intxt", self.path, 256, imgui.INPUT_TEXT_ALWAYS_OVERWRITE)
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("...", 26, 19):
                selected_file = filedialog.askopenfilename()
                if selected_file is not None:
                    if get_filetype(selected_file) in ['.tiff', '.tif']:
                        self.path = selected_file
                        self.on_select_file()
            imgui.columns(2, border = False)
            imgui.text("frames:")
            imgui.text("image size:")
            imgui.text("pixel size:  ")
            imgui.next_column()
            imgui.new_line()
            imgui.same_line(spacing=3)
            imgui.text(f"{self.dataset.n_frames}")
            imgui.new_line()
            imgui.same_line(spacing=3)
            imgui.text(f"{self.dataset.img_width}x{self.dataset.img_height}")
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *Node.COLOUR_WINDOW_BACKGROUND)
            imgui.push_item_width(35)
            _, self.pixel_size = imgui.input_float("##nm", self.pixel_size, 0.0, 0.0, format = "%.1f")
            imgui.pop_item_width()
            imgui.same_line()
            imgui.text("nm")
            imgui.pop_style_color(1)
            imgui.columns(1)

            _, self.load_on_the_fly = imgui.checkbox("Load on the fly", self.load_on_the_fly)
            if not self.load_on_the_fly:
                imgui.text("Loading progress:")
                width = imgui.get_content_region_available_width()
                origin = imgui.get_window_position()
                y = imgui.get_cursor_screen_pos()[1]
                drawlist = imgui.get_window_draw_list()
                drawlist.add_rect_filled(Node.PROGRESS_BAR_PADDING + origin[0], y,
                                         Node.PROGRESS_BAR_PADDING + origin[0] + width,
                                         y + Node.PROGRESS_BAR_HEIGHT,
                                         imgui.get_color_u32_rgba(*Node.colour_whiten(Node.COLOUR[self.type])))
                drawlist.add_rect_filled(Node.PROGRESS_BAR_PADDING + origin[0], y, Node.PROGRESS_BAR_PADDING + origin[0] + width * min([(self.to_load_idx / self.n_to_load), 1.0]), y + Node.PROGRESS_BAR_HEIGHT, imgui.get_color_u32_rgba(*Node.COLOUR[self.type]))

            super().render_end()

    def on_select_file(self):
        self.dataset = Dataset(self.path)
        self.dataset.pixel_size = self.pixel_size
        self.n_to_load = self.dataset.n_frames
        self.done_loading = False
        self.to_load_idx = 0
        self.any_change = True
        NodeEditor.set_active_node(self)

    def get_image_impl(self, idx):
        if self.dataset.n_frames > 0:
            retimg = self.dataset.get_indexed_image(idx)
            retimg.clean()
            return retimg
        else:
            return None

    def on_update(self):
        if not self.load_on_the_fly and not self.done_loading:
            if self.to_load_idx < self.dataset.n_frames:
                self.dataset.get_indexed_image(self.to_load_idx).load()
                self.to_load_idx += 1
            else:
                self.done_loading = True


class RegisterNode(Node):
    METHODS = ["TurboReg", "ORB (todo)"]
    REFERENCES = ["Input image", "Template frame", "Consecutive pairing"]

    def __init__(self):
        super().__init__(Node.TYPE_REGISTER)  # Was: super(LoadDataNode, self).__init__()
        self.size = [230, 185]

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.image_in = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.INPUT, parent = self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)
        self.connectable_attributes.append(self.image_in)

        # Set up node-specific vars
        self.register_method = 0
        self.reference_method = 1
        self.reference_image = None
        self.frame = 0
        self.roi = [0, 0, 0, 0]
        # StackReg vars
        self.sr = StackReg(StackReg.TRANSLATION)

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            _c, self.use_roi = imgui.checkbox("use ROI", self.use_roi)
            self.any_change = self.any_change or _c
            _method_changed, self.register_method = imgui.combo("Method", self.register_method, RegisterNode.METHODS)
            _reference_changed, self.reference_method = imgui.combo("Reference", self.reference_method, RegisterNode.REFERENCES)
            _frame_changed = False
            if self.reference_method == 1:
                imgui.push_item_width(50)
                _frame_changed, self.frame = imgui.input_int("Template frame", self.frame, 0, 0)
                imgui.pop_item_width()
            if self.reference_method == 0:
                imgui.new_line()
                self.image_in.render_start()
                self.image_in.render_end()

            self.any_change = self.any_change or _method_changed or _reference_changed or _frame_changed

            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            input_img = data_source.get_image(idx)
            reference_img = None
            # Get reference frame according to specified pairing method
            if self.reference_method == 2:
                reference_img = data_source.get_image(idx - 1)
            elif self.reference_method == 0:
                reference_img = self.image_in.get_incoming_node().get_image(idx=None)
            elif self.reference_method == 1:
                reference_img = data_source.get_image(self.frame)

            # Perform registration according to specified registration method
            if self.register_method == 0:
                if reference_img is not None:
                    template = reference_img.load()
                    image = input_img.load()
                    if self.use_roi:
                        template = template[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                        image = image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                    tmat = self.sr.register(template, image)
                    input_img.translation = [tmat[0][2], tmat[1][2]]
            else:
                NodeEditor.set_error(Exception(), "RegisterNode: reference method not available (may not be implemented yet).")

            input_img.bake_transform()
            return input_img


class GetImageNode(Node):
    IMAGE_MODES = ["By frame", "Time projection"]
    PROJECTIONS = ["Average", "Minimum", "Maximum", "St. dev."]

    def __init__(self):
        super().__init__(Node.TYPE_GET_IMAGE)
        self.size = [200, 120]
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT,parent=self)
        self.image_out = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.OUTPUT, parent=self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.image_out)

        self.mode = 0
        self.projection = 0
        self.frame = 0
        self.image = None
        self.load_data_source = None

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.image_out.render_start()
            self.dataset_in.render_end()
            self.image_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.push_item_width(140)
            _c, self.mode = imgui.combo("Mode", self.mode, GetImageNode.IMAGE_MODES)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(80)
            if self.mode == 0:
                _c, self.frame = imgui.input_int("Frame nr.", self.frame, 0, 0)
                self.any_change = self.any_change or _c
            elif self.mode == 1:
                _c, self.projection = imgui.combo("Projection", self.projection, GetImageNode.PROJECTIONS)
                if _c:
                    self.image = None
                self.any_change = self.any_change or _c
            imgui.pop_item_width()
            if self.any_change:
                self.configure_settings()

            super().render_end()

    def configure_settings(self):
        datasource = self.dataset_in.get_incoming_node()
        if datasource:
            try:
                if self.mode == 0:
                    self.image = datasource.get_image(self.frame).load()
                elif self.mode == 1:
                    load_data_node = Node.get_source_load_data_node(self)
                    load_data_node.load_on_the_fly = False
                    self.load_data_source = load_data_node
            except Exception as e:
                NodeEditor.set_error(e, "GetImageNode error upon attempting to gen img.\n"+str(e))
        else:
            NodeEditor.set_error(Exception(), "GetImageNode missing input dataset.")
        self.any_change = True

    def on_update(self):
        if self.mode == 1 and self.image is None:
            if self.load_data_source is not None:
                if self.load_data_source.done_loading:
                    self.generate_projection()

    def generate_projection(self):
        data_source = self.dataset_in.get_incoming_node()
        frame = data_source.get_image(0)
        n_frames = Node.get_source_load_data_node(self).dataset.n_frames
        projection_image = np.zeros((frame.width, frame.height, n_frames))
        for i in range(n_frames):
            projection_image[:, :, i] = data_source.get_image(i).load()
        if self.projection == 0:
            self.image = np.average(projection_image, axis = 2)
        elif self.projection == 1:
            self.image = np.min(projection_image, axis = 2)
        elif self.projection == 2:
            self.image = np.max(projection_image, axis = 2)
        elif self.projection == 3:
            self.image = np.std(projection_image, axis = 2)
        self.any_change = True

    def get_image_impl(self, idx=None):
        if self.any_change:
            self.configure_settings()
        if self.image is not None:
            virtual_frame = Frame("virtual_frame")
            virtual_frame.data = self.image
            return virtual_frame


class ImageCalculatorNode(Node):
    ## Note: the output dataset has all the metadata of dataset_in_a
    OPERATIONS = ["Add", "Subtract", "Divide", "Multiply"]

    def __init__(self):
        super().__init__(Node.TYPE_IMAGE_CALCULATOR)
        self.size = [230, 105]
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, parent=self, allowed_partner_types=[ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.TYPE_IMAGE])
        self.input_b = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, parent=self, allowed_partner_types=[ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.TYPE_IMAGE])
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)
        self.image_out = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.OUTPUT, parent=self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.input_b)
        self.connectable_attributes.append(self.dataset_out)
        self.connectable_attributes.append(self.image_out)

        self.operation = 1

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            if self.dataset_in.current_type == ConnectableAttribute.TYPE_IMAGE:
                self.image_out.render_start()
                self.image_out.render_end()
            else:
                self.dataset_out.render_start()
                self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            self.input_b.render_start()
            self.input_b.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.push_item_width(90)
            _c, self.operation = imgui.combo("Operation", self.operation, ImageCalculatorNode.OPERATIONS)
            self.any_change = self.any_change | _c
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        try:
            source_a = self.dataset_in.get_incoming_node()
            source_b = self.input_b.get_incoming_node()
            if source_a and source_b:
                img_a = source_a.get_image(idx)
                img_b = source_b.get_image(idx)
                img_a_pxd = img_a.load()
                img_b_pxd = img_b.load()
                img_out = None
                if self.operation == 0:
                    img_out = img_a_pxd + img_b_pxd
                elif self.operation == 1:
                    img_out = img_a_pxd - img_b_pxd
                elif self.operation == 2:
                    img_out = img_a_pxd / img_b_pxd
                elif self.operation == 3:
                    img_out = img_a_pxd * img_b_pxd
                if self.dataset_in.current_type != ConnectableAttribute.TYPE_IMAGE:
                    img_a.data = img_out
                    return img_a
                else:
                    virtual_frame = Frame("virtual_frame")
                    virtual_frame.data = img_out.astype(np.uint16)
                    return virtual_frame
        except Exception as e:
            NodeEditor.set_error(Exception(), "ImageCalculatorNode error:\n"+str(e))


class SpatialFilterNode(Node):
    FILTERS = ["Wavelet", "Gaussian", "Median"]
    WAVELETS = dict()
    WAVELETS["Haar"] = 'haar'
    WAVELETS["Symlet 2"] = 'sym2'
    WAVELETS["Symlet 3"] = 'sym3'
    WAVELETS["Daubechies 2"] = 'db2'
    WAVELETS["Biorthogonal 1.3"] = 'bior1.3'
    WAVELETS["Reverse biorthogonal 2.2"] = 'rbio2.2'
    WAVELETS["Other..."] = None


    WAVELET_NAMES = list(WAVELETS.keys())
    WAVELET_OTHER_IDX = WAVELET_NAMES.index("Other...")
    def __init__(self):
        super().__init__(Node.TYPE_SPATIAL_FILTER)  # Was: super(LoadDataNode, self).__init__()
        self.size = [210, 130]

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)
        # Set up node-specific vars
        self.filter = 1
        self.level = 1
        self.sigma = 2.0
        self.kernel = 3
        self.wavelet = 0
        self.custom_wavelet = "bior6.8"

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.push_item_width(140)
            _c, self.filter = imgui.combo("Filter", self.filter, SpatialFilterNode.FILTERS)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(70)
            if self.filter == 0:
                imgui.push_item_width(140)
                _c, self.wavelet = imgui.combo("Wavelet", self.wavelet, SpatialFilterNode.WAVELET_NAMES)
                imgui.pop_item_width()
                self.any_change = self.any_change or _c
                if self.wavelet == SpatialFilterNode.WAVELET_OTHER_IDX:
                    _c, self.custom_wavelet = imgui.input_text("pywt name", self.custom_wavelet, 16)
                    self.any_change = self.any_change or _c
                _c, self.level = imgui.input_int("Level", self.level, 0, 0)
                self.any_change = self.any_change or _c
            elif self.filter == 1:
                _c, self.sigma = imgui.input_float("Sigma (px)", self.sigma, 0.0, 0.0, format="%.1f")
                self.any_change = self.any_change or _c
            elif self.filter == 2:
                _c, self.kernel = imgui.input_int("Kernel (px)", self.kernel, 0, 0)
                if self.kernel % 2 == 0:
                    self.kernel += 1
                self.any_change = self.any_change or _c
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            input_image = data_source.get_image(idx)
            output_image = Frame("virtual_path")
            if self.filter == 0:
                chosen_wavelet = SpatialFilterNode.WAVELETS[SpatialFilterNode.WAVELET_NAMES[self.wavelet]]
                if chosen_wavelet is None:
                    chosen_wavelet = self.custom_wavelet
                in_pxd = input_image.load()
                output_image.data= pywt.swt2(in_pxd, wavelet=chosen_wavelet, level=self.level, norm=True, trim_approx=True)[0]
            elif self.filter == 1:
                output_image.data = gaussian_filter(input_image.load(), self.sigma)
            elif self.filter == 2:
                output_image.data = medfilt(input_image.load(), self.kernel)
            return output_image
        else:
            return None


class TemporalFilterNode(Node):
    FILTERS = ["Forward difference", "Backward difference", "Central difference", "Grouped difference", "Windowed average"]
    NEGATIVE_MODES = ["Absolute", "Zero", "Retain"]
    INCOMPLETE_GROUP_MODES = ["Discard"]

    def __init__(self):
        super().__init__(Node.TYPE_TEMPORAL_FILTER)  # Was: super(LoadDataNode, self).__init__()
        self.size = [250, 220]

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)

        self.filter = 0
        self.negative_handling = 1
        self.incomplete_group_handling = 0
        self.skip = 1
        self.group_size = 11
        self.group_background_index = 1
        self.window = 3

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(170)
            _c, self.filter = imgui.combo("Filter", self.filter, TemporalFilterNode.FILTERS)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(110)
            _c, self.negative_handling = imgui.combo("Negative handling", self.negative_handling, TemporalFilterNode.NEGATIVE_MODES)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(90)
            if self.filter == 0 or self.filter == 1 or self.filter == 2:
                _c, self.skip = imgui.input_int("Step (# frames)", self.skip, 0, 0)
                self.any_change = self.any_change or _c
            elif self.filter == 3:
                _c, self.group_size = imgui.input_int("Images per cycle", self.group_size, 0, 0)
                self.any_change = self.any_change or _c
                _c, self.group_background_index = imgui.input_int("Background index", self.group_background_index, 0, 0)
                self.any_change = self.any_change or _c
                _c, self.incomplete_group_handling = imgui.combo("Incomplete groups", self.incomplete_group_handling, TemporalFilterNode.INCOMPLETE_GROUP_MODES)
                self.any_change = self.any_change or _c
            elif self.filter == 4:
                _c, self.window = imgui.input_int("Window size", self.window, 0, 0)
                self.any_change = self.any_change or _c
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            pxd = None
            if self.filter == 0:
                pxd = data_source.get_image(idx + self.skip).load() - data_source.get_image(idx).load()
            elif self.filter == 1:
                pxd = data_source.get_image(idx).load() - data_source.get_image(idx - self.skip).load()
            elif self.filter == 2:
                pxd = data_source.get_image(idx + self.skip).load() - data_source.get_image(idx - self.skip).load()
            elif self.filter == 3:
                pxd = data_source.get_image(idx // self.group_size).load() - data_source.get_image(self.group_size * (idx // self.group_size) + self.group_background_index).load()
            elif self.filter == 4:
                pxd = np.zeros_like(data_source.get_image(idx).load())
                for i in range(-self.window, self.window + 1):
                    pxd += data_source.get_image(idx + i).load()
                pxd /= (2 * self.window + 1)

            if self.negative_handling == 0:
                pxd = np.abs(pxd)
            elif self.negative_handling == 1:
                pxd[pxd < 0] = 0
            elif self.negative_handling == 2:
                pass

            outframe = Frame("virtual_frame")
            outframe.data = pxd
            return outframe


class FrameShiftNode(Node):
    def __init__(self):
        super().__init__(Node.TYPE_FRAME_SHIFT)
        self.size = [150, 100]

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)

        self.shift = 0

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(80)
            _c, self.shift = imgui.input_int("shift", self.shift, 1, 10)
            self.any_change = _c or self.any_change

            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            return data_source.get_image(idx + self.shift)


class FrameSelectionNode(Node):
    def __init__(self):
        super().__init__(Node.TYPE_FRAME_SELECTION)  # Was: super(LoadDataNode, self).__init__()
        self.size = [200, 200]

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent = self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent = self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            super().render_end()


class ReconstructionRendererNode(Node):
    COLOUR_MODE = ["RGB, LUT"]
    def __init__(self):
        super().__init__(Node.TYPE_RECONSTRUCTOR)  # Was: super(LoadDataNode, self).__init__()
        self.size = [250, 200]

        self.reconstruction_in = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.INPUT, parent = self)
        self.image_out = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.OUTPUT, parent = self)
        self.connectable_attributes.append(self.reconstruction_in)
        self.connectable_attributes.append(self.image_out)

        self.reconstructor = Reconstructor()
        self.latest_image = None
        self.path = ""
        self.particledata = None
        self.pixel_size_in = 64
        self.magnification = 10

    def render(self):
        if super().render_start():
            any_change = False
            self.reconstruction_in.render_start()
            self.image_out.render_start()
            any_change = self.reconstruction_in.render_end() | any_change
            any_change = self.image_out.render_end()| any_change

            imgui.spacing()
            imgui.separator()
            imgui.spacing()


            if not self.reconstruction_in.is_connected():
                imgui.text("Load reconstruction .csv")
                imgui.push_item_width(150)
                _field_changed, self.path = imgui.input_text("##incsv", self.path, 256)
                any_change = any_change | _field_changed
                imgui.pop_item_width()
                imgui.same_line()
                if imgui.button("...", 26, 19):
                    selected_file = filedialog.askopenfilename()
                    if selected_file is not None:
                        if get_filetype(selected_file) == ".csv":
                            self.path = selected_file
                            self.reconstructor.set_particle_data(self.path)
                            any_change = True
                        else:
                            NodeEditor.set_error(Exception(), "Reconstruction data file must be of type '.csv'")
            imgui.push_item_width(70)
            _pixel_in_changed, self.pixel_size_in = imgui.input_float("Raw pixel size (nm)", self.pixel_size_in, 0, 0, format = "%.1f")
            _mag_changed, self.magnification = imgui.input_float("Magnification", self.magnification, 0, 0, format = "%.1f")
            if _pixel_in_changed or _mag_changed:
                self.play = False
            imgui.pop_item_width()
            if _mag_changed or _pixel_in_changed:
                self.reconstructor.set_pixel_size(self.pixel_size_in / self.magnification)

            super().render_end()
            any_change = any_change | self.play_button()
            if any_change and self.play:
                try:
                    self.latest_image = self.reconstructor.render()
                    self.any_change = True
                except Exception as e:
                    NodeEditor.set_error(e, str(e))

    def get_image_impl(self, idx=None):
        if self.latest_image is not None:
            img_wrapper = Frame("virtual_path")
            img_wrapper.data = self.latest_image
            return img_wrapper
        else:
            return None


class ParticlePainterNode(Node):
    PARAMETERS = ["Frame", "x Position", "y Position", "Sigma", "Offset", "Uncertainty", "Background std."]
    PAINTS = ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta"]
    PAINT_COLOURS = [
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0)
    ]

    def __init__(self):
        super().__init__(Node.TYPE_PARTICLE_PAINTER)  # Was: super(LoadDataNode, self).__init__()
        self.size = [270, 240]

        self.reconstruction_in = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.INPUT, parent = self)
        self.reconstruction_out = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.OUTPUT, parent = self)

        self.connectable_attributes.append(self.reconstruction_in)
        self.connectable_attributes.append(self.reconstruction_out)

        self.parameter = 0
        self.paint = 0
        self.histogram_values = np.asarray([0, 0]).astype('float32')
        self.histogram_bins = np.asarray([0, 0]).astype('float32')
        self.min = 0
        self.max = 0

    def render(self):
        if super().render_start():
            self.reconstruction_in.render_start()
            self.reconstruction_out.render_start()
            self.reconstruction_in.render_end()
            self.reconstruction_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _, self.parameter = imgui.combo("Parameter", self.parameter, ParticlePainterNode.PARAMETERS)
            _, self.paint = imgui.combo("Paint", self.paint, ParticlePainterNode.PAINTS)
            imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *ParticlePainterNode.PAINT_COLOURS[self.paint])
            imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *ParticlePainterNode.PAINT_COLOURS[self.paint])
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *ParticlePainterNode.PAINT_COLOURS[self.paint])
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, *ParticlePainterNode.PAINT_COLOURS[self.paint])
            content_width = imgui.get_content_region_available_width()
            imgui.plot_histogram("##hist", self.histogram_values, graph_size = (content_width, 40))
            imgui.text("{:.2f}".format(self.histogram_bins[0]))
            imgui.same_line(position = content_width - imgui.get_font_size() * len(str(self.histogram_bins[-1])) / 2)
            imgui.text("{:.2f}".format(self.histogram_bins[-1]))
            imgui.push_item_width(content_width)
            _, self.min = imgui.slider_float("##min", self.min, self.histogram_bins[0], self.histogram_bins[-1], format = "min: %1.2f")
            _, self.max = imgui.slider_float("##max", self.max, self.histogram_bins[0], self.histogram_bins[-1], format="max: %1.2f")
            imgui.pop_style_color(4)
            imgui.pop_item_width()
            super().render_end()


class ParticleDetectionNode(Node):
    METHODS = ["Local maximum"]
    THRESHOLD_OPTIONS = ["Value", "St. Dev.", "Mean"]

    def __init__(self):
        super().__init__(Node.TYPE_PARTICLE_DETECTION)
        self.size = [290, 205]

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)

        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)

        self.method = 0
        self.thresholding = 1
        self.threshold = 100
        self.sigmas = 2.0
        self.means = 3.0
        self.n_max = 100
        self.d_min = 1

        self.roi = [0, 0, 0, 0]

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.dataset_out.render_start()
            self.dataset_in.render_end()
            self.dataset_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _c, self.use_roi = imgui.checkbox("use ROI", self.use_roi)
            self.any_change = self.any_change or _c
            imgui.push_item_width(160)
            _c, self.method = imgui.combo("Detection method", self.method, ParticleDetectionNode.METHODS)
            self.any_change = self.any_change or _c
            _c, self.thresholding = imgui.combo("Threshold method", self.thresholding, ParticleDetectionNode.THRESHOLD_OPTIONS)
            if self.thresholding == 0:
                _c, self.threshold = imgui.input_int("Threshold level", self.threshold, 0, 0)
                self.any_change = self.any_change or _c
            elif self.thresholding == 1:
                _c, self.sigmas = imgui.slider_float("Sigmas", self.sigmas, 0.1, 10.0, format = "%.1f")
                self.any_change = self.any_change or _c
            elif self.thresholding == 2:
                _c, self.means = imgui.slider_float("Means", self.means, 0.1, 10.0, format = "%.1f")
                self.any_change = self.any_change or _c
            imgui.pop_item_width()
            imgui.push_item_width(70)
            _c, self.n_max = imgui.input_int("Max. # of particles", self.n_max, 0, 0)
            self.any_change = self.any_change or _c
            _c, self.d_min = imgui.input_int("Minimum distance (px)", self.d_min, 0, 0)
            self.any_change = self.any_change or _c
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        source = self.dataset_in.get_incoming_node()
        if source is not None:
            # Find threshold value
            image_obj = source.get_image(idx)
            image = image_obj.load()
            if self.use_roi:
                image = image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
            threshold = self.threshold
            if self.thresholding == 1:
                threshold = self.sigmas * np.std(image)
            if self.thresholding == 2:
                threshold = self.means * np.mean(image)
            # Perform requested detection method
            coordinates = peak_local_max(image, threshold_abs = threshold, num_peaks = self.n_max, min_distance = self.d_min) + np.asarray([self.roi[1], self.roi[0]])

            image_obj.maxima = coordinates
            return image_obj


class ExportDataNode(Node):

    def __init__(self):
        super().__init__(Node.TYPE_EXPORT_DATA)
        self.size = [210, 220]

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT,
                                               parent=self)
        self.image_in = ConnectableAttribute(ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.image_in)

        self.path = "..."
        self.roi = [0, 0, 0, 0]
        self.include_discarded_frames = False
        self.saving = False
        self.export_type = 0  # 0 for dataset, 1 for image.

        self.frames_to_load = list()
        self.n_jobs = 1
        self.n_frames_to_save = 1
        self.n_frames_saved = 0

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.dataset_in.render_end()
            imgui.new_line()
            self.image_in.render_start()
            self.image_in.render_end()

            if self.image_in.newly_connected:
                self.export_type = 1
                self.dataset_in.disconnect_all()
            elif self.dataset_in.newly_connected:
                self.image_in.disconnect_all()
                self.export_type = 0

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            _, self.use_roi = imgui.checkbox("use ROI", self.use_roi)
            imgui.text("Output path")
            imgui.push_item_width(150)
            _, self.path = imgui.input_text("##intxt", self.path, 256, imgui.INPUT_TEXT_ALWAYS_OVERWRITE)
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("...", 26, 19):
                filename = filedialog.asksaveasfilename()
                if filename is not None:
                    if '.' in filename:
                        filename = filename[:filename.rfind(".")]
                    self.path = filename

            content_width = imgui.get_window_width()
            save_button_width = 100
            save_button_height = 40

            if self.export_type == 0:
                _c, self.include_discarded_frames = imgui.checkbox("Include discarded frames", self.include_discarded_frames)

            if self.saving:
                width = imgui.get_content_region_available_width()
                origin = imgui.get_window_position()
                y = imgui.get_cursor_screen_pos()[1]
                drawlist = imgui.get_window_draw_list()
                drawlist.add_rect_filled(Node.PROGRESS_BAR_PADDING + origin[0], y,
                                         Node.PROGRESS_BAR_PADDING + origin[0] + width,
                                         y + Node.PROGRESS_BAR_HEIGHT,
                                         imgui.get_color_u32_rgba(*Node.colour_whiten(Node.COLOUR[self.type])))
                drawlist.add_rect_filled(Node.PROGRESS_BAR_PADDING + origin[0], y,
                                         Node.PROGRESS_BAR_PADDING + origin[0] + width * self.n_frames_saved / self.n_frames_to_save, y + Node.PROGRESS_BAR_HEIGHT,
                                         imgui.get_color_u32_rgba(*Node.COLOUR[self.type]))

            imgui.spacing()
            imgui.spacing()
            imgui.spacing()
            imgui.new_line()
            imgui.same_line(position=(content_width - save_button_width) // 2)
            if not self.saving:
                if imgui.button("Save", save_button_width, save_button_height):
                    self.init_save()
            else:
                if imgui.button("Cancel", save_button_width, save_button_height):
                    self.saving = False
            super().render_end()

    def get_img_and_save(self, idx):
        img_pxd = self.get_image_impl(idx)
        if self.use_roi:
            img_pxd = img_pxd[self.roi[1]:self.roi[3],self.roi[0]:self.roi[2]]
        Image.fromarray(img_pxd).save(self.path+"/0"+str(idx)+".tif")

    def init_save(self):
        if self.export_type == 0:
            self.saving = True

            if self.include_discarded_frames:
                n_active_frames = Node.get_source_load_data_node(self).dataset.n_frames
                self.frames_to_load = list(range(0, n_active_frames))
            else:
                self.frames_to_load = list()
                for i in range(Node.get_source_load_data_node(self).dataset.n_frames):
                    self.frames_to_load.append(i)
            self.n_frames_to_save = len(self.frames_to_load)
            self.n_frames_saved = 0
            if not os.path.isdir(self.path):
                os.mkdir(self.path)

            self.n_jobs = cfg.n_cpus
            if cfg.n_cpus == -1:
                self.n_jobs = cpu_count()



        elif self.export_type == 1:
            img_pxd = self.get_image_impl(None).load()
            try:
                Image.fromarray(img_pxd).save(self.path+".tif")
            except Exception as e:
                NodeEditor.set_error(e, "Error saving image: "+str(e))

    def on_update(self):
        if self.saving:
            try:
                indices = list()
                for i in range(min([self.n_jobs, len(self.frames_to_load)])):
                    self.n_frames_saved += 1
                    indices.append(self.frames_to_load[-1])
                    self.frames_to_load.pop()

                Parallel(n_jobs=self.n_jobs, verbose = 10)(delayed(self.get_img_and_save)(index) for index in indices)
                if len(self.frames_to_load) == 0:
                    self.saving = False
            except Exception as e:
                self.saving = False
                NodeEditor.set_error(e, "Error saving stack: \n"+str(e))

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            incoming_img = data_source.get_image(idx)
            print(incoming_img)
            img_pxd = incoming_img.load()
            incoming_img.clean()
            return img_pxd
        img_source = self.image_in.get_incoming_node()
        if img_source:
            return img_source.get_image(idx)


class BinImageNode(Node):
    MODES = ["Average", "Median", "Min", "Max", "Sum"]

    def __init__(self):
        super().__init__(Node.TYPE_BIN_IMAGE)
        self.size = [170, 120]
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, self, [ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_DATASET])
        self.output = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.OUTPUT, self, [ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_DATASET])
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.output)

        self.factor = 2
        self.mode = 0

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.output.render_start()
            self.dataset_in.render_end()
            self.output.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()


            _c, self.mode = imgui.combo("Method", self.mode, BinImageNode.MODES)
            imgui.push_item_width(60)
            _c, self.factor = imgui.input_int("Bin factor", self.factor, 0, 0)
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            pxd = data_source.get_image(idx).load()
            width, height = pxd.shape
            pxd = pxd[:self.factor * (width // self.factor), :self.factor * (height // self.factor)]
            if self.mode == 0:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).mean(2).mean(0)
            elif self.mode == 1:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).mode(2).mode(0)
            elif self.mode == 2:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).min(2).min(0)
            elif self.mode == 3:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).max(2).max(0)
            elif self.mode == 4:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).sum(2).sum(0)
            virtual_frame = Frame("_")
            virtual_frame.data = pxd
            return virtual_frame


class ParticleFittingNode(Node):
    RANGE_OPTIONS = ["All frames", "Current frame only", "Range"]
    ESTIMATORS = ["Least squares", "Maximum likelihood", "No estimator"]
    PSFS = ["Gaussian", "Elliptical Gaussian"]

    def __init__(self):
        super().__init__(Node.TYPE_PARTICLE_FITTING)

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.reconstruction_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.reconstruction_out)

        self.size = [300, 260]
        self.range_option = 1
        self.range_min = 0
        self.range_max = 1
        self.estimator = 1
        self.crop_radius = 3
        self.initial_sigma = 1.6
        self.fitting = False
        self.n_to_fit = 1
        self.n_fitted = 0
        self.frames_to_fit = list()
        self.particles = list()

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.reconstruction_out.render_start()
            self.dataset_in.render_end()
            self.reconstruction_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _c, self.range_option = imgui.combo("##range_option_combo", self.range_option, ParticleFittingNode.RANGE_OPTIONS)
            self.any_change = _c or self.any_change
            if self.range_option == 2:
                imgui.push_item_width(80)
                _c, (self.range_min, self.range_max) = imgui.input_int2('(start, stop) index', self.range_min, self.range_max)
                self.any_change = _c or self.any_change
                imgui.pop_item_width()
            imgui.spacing()
            _c, self.estimator = imgui.combo("Estimator", self.estimator, ParticleFittingNode.ESTIMATORS)
            self.any_change = _c or self.any_change
            imgui.push_item_width(80)
            if self.estimator in [0, 1]:
                _c, self.initial_sigma = imgui.input_float("Initial sigma (px)", self.initial_sigma, 0, 0, "%.1f")
                self.any_change = _c or self.any_change
                _c, self.crop_radius = imgui.input_int("Fitting radius (px)", self.crop_radius, 0, 0)
                self.any_change = _c or self.any_change
            imgui.pop_item_width()

            imgui.spacing()
            if self.fitting:
                width = imgui.get_content_region_available_width()
                origin = imgui.get_window_position()
                y = imgui.get_cursor_screen_pos()[1]
                drawlist = imgui.get_window_draw_list()
                drawlist.add_rect_filled(Node.PROGRESS_BAR_PADDING + origin[0], y, Node.PROGRESS_BAR_PADDING + origin[0] + width, y + Node.PROGRESS_BAR_HEIGHT, imgui.get_color_u32_rgba(*Node.colour_whiten(Node.COLOUR[self.type])))
                drawlist.add_rect_filled(Node.PROGRESS_BAR_PADDING + origin[0], y, Node.PROGRESS_BAR_PADDING + origin[0] + width * (self.n_fitted / self.n_to_fit), y + Node.PROGRESS_BAR_HEIGHT, imgui.get_color_u32_rgba(*Node.COLOUR[self.type]))

            if imgui.button("Fit", 50, 30):
                self.init_fit()


            super().render_end()

    def init_fit(self):
        try:
            self.particles = list()
            self.fitting = True
            self.frames_to_fit = list()
            if self.range_option == 0:
                dataset = self.get_source_load_data_node(self).dataset
                n_frames = dataset.n_frames
                self.frames_to_fit = list(range(0, n_frames))
            elif self.range_option == 2:
                self.frames_to_fit = list(range(self.range_min, self.range_max))
            elif self.range_option == 1:
                dataset = self.get_source_load_data_node(self).dataset
                self.frames_to_fit = [dataset.current_frame]
            self.n_to_fit = len(self.frames_to_fit)
            self.n_fitted = 0
        except Exception as e:
            NodeEditor.set_error(e, "Error in init_fit: "+str(e))


    def on_update(self):
        if self.fitting:
            if len(self.frames_to_fit) == 0:
                self.fitting = False
                print(self.particles)
                # TODO: make this node output self.particles, i.e. a reconstruction, in a format suitable for ReconstructionRendererNode
            else:
                fitted_frame = self.get_image_impl(self.frames_to_fit[-1])
                particles = fitted_frame.particles
                self.frames_to_fit.pop()
                self.particles += particles

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            frame = data_source.get_image(idx)
            particles = list()
            if self.estimator in [0, 1]:
                particles = pfit.frame_to_particles(frame, self.initial_sigma, self.estimator, self.crop_radius)
            elif self.estimator == 2:
                particles = list()
                pxd = frame.load()
                for i in range(frame.maxima.shape[0]):
                    intensity = pxd[frame.maxima[i, 0], frame.maxima[i, 1]]
                    x = frame.maxima[i, 0]
                    y = frame.maxima[i, 1]
                    particles.append(Particle(idx, x, y, 1.0, intensity))
            frame.particles = particles
            return frame





