import shutil
from itertools import count
import imgui
from imgui.integrations.glfw import GlfwRenderer
import config as cfg
import glfw
import time
import datetime
import util
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
import dill as pickle
import copy

tkroot = tk.Tk()
tkroot.withdraw()

_srnversion = "1.0.0"


class NodeEditor:
    if True:
        COLOUR_WINDOW_BACKGROUND = (0.94, 0.94, 0.94, 1.0)
        CONTEXT_MENU_SIZE = (200, 98)
        COLOUR_ERROR_WINDOW_BACKGROUND = (0.84, 0.84, 0.84, 1.0)
        COLOUR_ERROR_WINDOW_HEADER = (0.6, 0.3, 0.3, 1.0)
        COLOUR_ERROR_WINDOW_HEADER_NEW = (0.8, 0.35, 0.35, 1.0)
        COLOUR_ERROR_WINDOW_TEXT = (0.0, 0.0, 0.0, 1.0)
        COLOUR_MENU_WINDOW_BACKGROUND = (0.96, 0.96, 0.96, 1.0)
        COLOUR_CM_WINDOW_TEXT = (0.0, 0.0, 0.0, 1.0)
        COLOUR_CM_OPTION_HOVERED = (1.0, 1.0, 1.0, 1.0)

        COLOUR_MAIN_MENU_BAR = (0.98, 0.98, 0.98, 0.9)
        COLOUR_MAIN_MENU_BAR_TEXT = (0.0, 0.0, 0.0, 1.0)
        COLOUR_MAIN_MENU_BAR_HILIGHT = (0.96, 0.95, 0.92, 1.0)

        ERROR_WINDOW_HEIGHT = 80
    nodes = list()
    active_node = None
    focused_node = None
    next_active_node = None
    connectable_attributes = list()
    active_connector = None
    active_connector_parent_node = None
    active_connector_hover_pos = [0, 0]
    connector_released = False
    connector_delete_requested = False
    node_move_requested = [0, 0]
    camera_move_requested = [0, 0]

    error_msg = None
    error_new = True
    error_obj = None

    window_width = 1920
    window_height = 1080

    current_dataset_pixel_size = 100
    any_change = False

    profiling = False
    pickle_temp = None

    n_cpus_max = cpu_count()
    n_cpus = n_cpus_max
    batch_size = n_cpus_max

    def __init__(self, window, shared_font_atlas=None):
        self.window = window
        self.window.clear_color = NodeEditor.COLOUR_WINDOW_BACKGROUND
        self.window.make_current()

        if shared_font_atlas is not None:
            self.imgui_context = imgui.create_context(shared_font_atlas)
        else:
            # imgui.get_current_context()
            self.imgui_context = imgui.create_context()
        self.imgui_implementation = GlfwRenderer(self.window.glfw_window)
        self.window.set_mouse_callbacks()
        self.window.set_window_callbacks()

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
        NodeEditor.window_width = self.window.width
        NodeEditor.window_height = self.window.height

        for node in NodeEditor.nodes:
            node.clear_flags()
            node.on_update()
        if NodeEditor.focused_node is not None:
            NodeEditor.focused_node.any_change = NodeEditor.any_change
        NodeEditor.any_change = False
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
        imgui_want_mouse = imgui.get_io().want_capture_mouse
        if self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT) and imgui_want_mouse:
            NodeEditor.node_move_requested = self.window.cursor_delta
        elif self.window.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE) and not imgui_want_mouse:
            NodeEditor.camera_move_requested = self.window.cursor_delta

        imgui.new_frame()
        self._gui_main()
        imgui.render()
        self.imgui_implementation.render(imgui.get_draw_data())

    def end_frame(self):
        self.window.end_frame()

    def _gui_main(self):

        ## Render nodes  - render active_connector_parent_node first, to enable all other connectors' drop targets.
        source_node_id = -1
        if NodeEditor.active_connector_parent_node is not None:
            NodeEditor.active_connector_parent_node.render()
            source_node_id = NodeEditor.active_connector_parent_node.id
        for node in NodeEditor.nodes:
            if node.id is not source_node_id:
                node.render()

        ## Context menu
        if not self.context_menu_open:
            if self.window.get_mouse_event(glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, 0) and not imgui.get_io().want_capture_mouse:
                self.context_menu_position = self.window.cursor_pos
                self.context_menu_open = True
                self.context_menu_can_close = False
        else:
            self._context_menu()

        self._menu_bar()

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
            imgui.set_next_window_size(self.window.width, NodeEditor.ERROR_WINDOW_HEIGHT)
            imgui.set_next_window_position(0, self.window.height - NodeEditor.ERROR_WINDOW_HEIGHT)
            _, stay_open = imgui.begin("Warning", True, imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE)
            imgui.text(NodeEditor.error_msg)
            if imgui.button("(debug): raise error", 180, 20):
                raise NodeEditor.error_obj
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
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *NodeEditor.COLOUR_MENU_WINDOW_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_TEXT, *NodeEditor.COLOUR_CM_WINDOW_TEXT)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *NodeEditor.COLOUR_MENU_WINDOW_BACKGROUND)
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
                    new_node = Node.create_node_by_type(key)
            imgui.end_menu()
        if imgui.begin_menu("Image processing"):
            for key in [Node.TYPE_REGISTER, Node.TYPE_SPATIAL_FILTER, Node.TYPE_TEMPORAL_FILTER, Node.TYPE_FRAME_SELECTION, Node.TYPE_FRAME_SHIFT, Node.TYPE_BIN_IMAGE, Node.TYPE_IMAGE_CALCULATOR, Node.TYPE_GET_IMAGE, Node.TYPE_BAKE_STACK, Node.TYPE_CROP_IMAGE]:
                item_selected, _ = imgui.menu_item(Node.TITLE[key])
                if item_selected:
                    new_node = Node.create_node_by_type(key)
            imgui.end_menu()
        if imgui.begin_menu("Reconstruction nodes"):
            for key in [Node.TYPE_PARTICLE_DETECTION, Node.TYPE_PARTICLE_FITTING, Node.TYPE_PARTICLE_FILTER, Node.TYPE_PARTICLE_PAINTER, Node.TYPE_RECONSTRUCTOR]:
                item_selected, _ = imgui.menu_item(Node.TITLE[key])
                if item_selected:
                    new_node = Node.create_node_by_type(key)
            imgui.end_menu()
        if imgui.begin_menu("All nodes"):
            for key in Node.TITLE.keys():
                if key == Node.TYPE_NULL:
                    continue
                item_selected, _ = imgui.menu_item(Node.TITLE[key])
                if item_selected:
                    new_node = Node.create_node_by_type(key)
            imgui.end_menu()
        if new_node is not None:
            try:
                new_node.position = self.context_menu_position
                NodeEditor.set_active_node(new_node)
                self.context_menu_open = False
            except Exception as e:
                NodeEditor.set_error(e, "Error upon requesting new node (probably not implemented yet.)\nError:"+str(e))

        clear_active_node, _ = imgui.menu_item("Release focused node")
        if clear_active_node:
            NodeEditor.set_active_node(None, True)
        # End
        imgui.pop_style_color(3)
        imgui.end()

    def _menu_bar(self):
        imgui.push_style_color(imgui.COLOR_MENUBAR_BACKGROUND, *NodeEditor.COLOUR_MAIN_MENU_BAR)
        imgui.push_style_color(imgui.COLOR_TEXT, *NodeEditor.COLOUR_MAIN_MENU_BAR_TEXT)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *NodeEditor.COLOUR_MAIN_MENU_BAR_HILIGHT)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *NodeEditor.COLOUR_MAIN_MENU_BAR_HILIGHT)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *NodeEditor.COLOUR_MENU_WINDOW_BACKGROUND)
        ## Save node setup.
        if imgui.core.begin_main_menu_bar():
            if imgui.begin_menu('File'):
                load_setup, _ = imgui.menu_item("Load node setup")
                if load_setup:
                    try:
                        filename = filedialog.askopenfilename(filetypes=[("srNodes setup", ".srn")])
                        if filename != '':
                            with open(filename, 'rb') as pickle_file:
                                imported_nodes = pickle.load(pickle_file)
                                NodeEditor.nodes = imported_nodes
                                NodeEditor.relink_after_load()
                    except Exception as e:
                        NodeEditor.set_error(e, f"Error loading node setup - are you sure you selected a '.srn' file?\n"+str(e))
                import_setup, _ = imgui.menu_item("Append node setup")
                if import_setup:
                    try:
                        filename = filedialog.askopenfilename(filetypes=[("srNodes setup", ".srn")])
                        if filename != '':
                            with open(filename, 'rb') as pickle_file:
                                imported_nodes = pickle.load(pickle_file)
                                for node in imported_nodes:
                                    NodeEditor.nodes.append(node)
                                NodeEditor.relink_after_load()
                    except Exception as e:
                        NodeEditor.set_error(e, "Error importing node setup - are you sure you selected a '.srn' file?\n"+str(e))
                save_setup, _ = imgui.menu_item("Save current setup")
                if save_setup:
                    filename = filedialog.asksaveasfilename(filetypes=[("srNodes setup", ".srn")])
                    if filename != '':
                        if filename[-4:] == ".srn":
                            filename = filename[:-4]
                        with open(filename+".srn", 'wb') as pickle_file:
                            node_copies = list()
                            for node in NodeEditor.nodes:
                                node.pre_save()
                                node_copies.append(copy.deepcopy(node))
                                node.post_save()
                            pickle.dump(node_copies, pickle_file)
                clear_setup, _ = imgui.menu_item("Clear current setup")
                if clear_setup:
                    for i in range(len(NodeEditor.nodes)):
                        NodeEditor.nodes[0].delete()
                    NodeEditor.nodes = list()
                imgui.end_menu()
            if imgui.begin_menu('Settings'):
                if imgui.begin_menu('Profiling'):
                    _c, NodeEditor.profiling = imgui.checkbox("Track node processing times", NodeEditor.profiling)
                    Node.tooltip("Keep track of the time that every node in the pipeline takes to process and output a frame.\n")
                    clear_times, _ = imgui.menu_item("   Reset timers")
                    if clear_times:
                        for node in NodeEditor.nodes:
                            node.profiler_time = 0.0
                            node.profiler_count = 0
                    imgui.end_menu()
                imgui.set_next_item_width(30)
                _c, NodeEditor.n_cpus = imgui.input_int("Parallel batch size", NodeEditor.n_cpus, 0, 0)
                if _c:
                    NodeEditor.n_cpus = max([NodeEditor.n_cpus, 1])
                    NodeEditor.batch_size = NodeEditor.n_cpus

                Node.tooltip("Number of frames to process within one parallel processing batch. Values higher than the amount\n" 
                             "of CPUs on the PC are allowed and will result in multiple tasks being dispatched to individual\n" 
                             "CPUs per batch. This can increase processing speed, but reduces GUI responsiveness. For optimal\n" 
                             "efficiency, set the batch size to an integer multiple of the amount of CPUs on the machine. \n"
                             f"This PC has: {NodeEditor.n_cpus_max} CPUs.")
                imgui.end_menu()
            imgui.end_main_menu_bar()
        imgui.pop_style_color(5)

    @staticmethod
    def set_active_node(node, keep_active=False):
        """
        :param node: Node type object or None.
        :param keep_active: bool. If True, the node is not just set as active_node, but also as the focused_node - this means that the node is kept in focus until it is manually unfocused by the user (in node right click context menu).
        :return: False if no change; i.e. when input node is already the active node. True otherwise.
        """
        if keep_active:
            NodeEditor.focused_node = node
            NodeEditor.next_active_node = node
            return True
        elif NodeEditor.focused_node is None:
            if NodeEditor.active_node is not None:
                if NodeEditor.active_node == node:
                    return False
            if node is None:
                NodeEditor.active_node = None
                return True
            else:
                NodeEditor.next_active_node = node
                if node.type == Node.TYPE_LOAD_DATA:
                    NodeEditor.current_dataset_pixel_size = node.pixel_size
                return True
        else:
            return False

    @staticmethod
    def set_error(error_obj, error_msg):
        NodeEditor.error_msg = error_msg
        NodeEditor.error_obj = error_obj
        NodeEditor.error_new = True

    @staticmethod
    def relink_after_load():
        nodes = NodeEditor.nodes
        attributes = list()
        ids_to_link = list()
        for node in nodes:
            for attribute in node.connectable_attributes:
                attributes.append(attribute)
                if attribute.direction == ConnectableAttribute.INPUT:
                    for partner in attribute.linked_attributes:
                        ids_to_link.append((attribute.id, partner.id))
                attribute.disconnect_all()

        def _find_attribute_by_id(target_id):
            for a in attributes:
                if a.id == target_id:
                    return a

        for pair in ids_to_link:
            ca_a = _find_attribute_by_id(pair[0])
            ca_b = _find_attribute_by_id(pair[1])
            ca_a.connect_attributes(ca_b)

    @staticmethod
    def delete_temporary_files():
        dirs = glob.glob("_srnodes_temp*/")
        for dir in dirs:
            shutil.rmtree(dir)


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
        TYPE_BAKE_STACK = 16
        TYPE_CROP_IMAGE = 17

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
        COLOUR[TYPE_BAKE_STACK] = (143 / 255, 123 / 255, 103 / 255, 1.0)
        COLOUR[TYPE_CROP_IMAGE] = (143 / 255, 123 / 255, 103 / 255, 1.0)

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
        TITLE[TYPE_BAKE_STACK] = "Bake stack"
        TITLE[TYPE_CROP_IMAGE] = "Crop image"
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

    def __init__(self, nodetype):
        self.version = _srnversion
        self.id = int(datetime.datetime.utcnow().timestamp()) + next(Node.id_generator)
        self.type = nodetype
        self.position = [0, 0]
        self.last_measured_window_position = [0, 0]
        self.size = [0, 0]
        self.connectable_attributes = list()
        self.colour = Node.COLOUR[self.type]
        self.title = Node.TITLE[self.type]
        self.play = False
        self.any_change = False
        self.queued_actions = list()
        self.use_roi = False
        if self.type is not Node.TYPE_NULL:
            NodeEditor.nodes.append(self)

        # bookkeeping
        self.buffer_last_output = False
        self.last_index_requested = -1
        self.last_frame_returned = None
        self.context_menu_open = False
        self.context_menu_position = [0, 0]
        self.context_menu_can_close = False
        self.keep_active = False
        # flags
        self.returns_image = True
        self.does_profiling_time = True
        self.does_profiling_count = True
        self.profiler_time = 0.0
        self.profiler_count = 0
        self.frame_requested_by_image_viewer = False

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False

    def clear_flags(self):
        for a in self.connectable_attributes:
            a.clear_flags()
        if self.any_change and not NodeEditor.focused_node == self:
            NodeEditor.any_change = True
        self.any_change = False

    def render_start(self):
        ## render the node window
        imgui.set_next_window_size(*self.size, imgui.ONCE)
        imgui.set_next_window_position(self.position[0], self.position[1], imgui.ALWAYS)
        imgui.set_next_window_size_constraints((self.size[0], 50), (self.size[0], 1000))
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *Node.COLOUR[self.type])
        imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, *Node.COLOUR[self.type])
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *Node.COLOUR[self.type])
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *Node.COLOUR[self.type])
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_COLLAPSED, *Node.COLOUR[self.type])
        imgui.push_style_color(imgui.COLOR_CHECK_MARK, *Node.COLOUR[self.type])
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *Node.COLOUR[self.type])
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *self.colour_brighten(Node.COLOUR[self.type]))
        if NodeEditor.focused_node == self:
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *self.colour_whiten(Node.COLOUR[self.type]))
            imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, Node.FOCUSED_NODE_BORDER_THICKNESS)
            imgui.push_style_color(imgui.COLOR_BORDER, *Node.COLOUR_WINDOW_BORDER_FOCUSED_NODE)
            imgui.push_style_color(imgui.COLOR_BORDER_SHADOW, *Node.COLOUR_WINDOW_BORDER_FOCUSED_NODE)
        elif NodeEditor.active_node == self:
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
        imgui.push_style_color(imgui.COLOR_HEADER, *self.colour_whiten(Node.COLOUR[self.type]))
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *self.colour_whiten(Node.COLOUR[self.type]))
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *self.colour_whiten(Node.COLOUR[self.type]))
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *self.colour_whiten(Node.COLOUR[self.type]))
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, *Node.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_BUTTON, *Node.COLOUR[self.type])
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *self.colour_brighten(Node.COLOUR[self.type]))
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *self.colour_whiten(Node.COLOUR[self.type]))
        imgui.push_style_color(imgui.COLOR_TAB_HOVERED, *self.colour_brighten(Node.COLOUR[self.type]))
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, Node.WINDOW_ROUNDING)
        imgui.push_style_var(imgui.STYLE_GRAB_ROUNDING, Node.WINDOW_ROUNDING)
        imgui.push_style_var(imgui.STYLE_FRAME_ROUNDING, Node.FRAME_ROUNDING)
        imgui.push_style_var(imgui.STYLE_WINDOW_MIN_SIZE, (1, 1))
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *Node.COLOUR_FRAME_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *Node.COLOUR_WINDOW_BACKGROUND)
        _, stay_open = imgui.begin(self.title + f"##{self.id}", True, imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_COLLAPSE)
        self.size[1] = imgui.get_window_size()[1]
        if imgui.is_window_focused() and not imgui.is_any_item_hovered():
            if NodeEditor.set_active_node(self):
                self.on_gain_focus()
            self.position[0] = self.position[0] + NodeEditor.node_move_requested[0]
            self.position[1] = self.position[1] + NodeEditor.node_move_requested[1]
        self.position[0] += NodeEditor.camera_move_requested[0]
        self.position[1] += NodeEditor.camera_move_requested[1]
        self.last_measured_window_position = imgui.get_window_position()
        imgui.push_clip_rect(0, 0, NodeEditor.window_width, NodeEditor.window_height)
        imgui.push_id(str(self.id))
        if not stay_open:
            self.render_end()
            self.delete()
            return False

        return True

    def render(self):
        pass

    def render_end(self):
        ## Node context_menu:
        if imgui.begin_popup_context_window():
            self.keep_active = NodeEditor.focused_node == self
            _changed, self.keep_active = imgui.checkbox("Focus node", self.keep_active)
            if _changed:
                if self.keep_active:
                    NodeEditor.set_active_node(self, True)
                else:
                    NodeEditor.set_active_node(None, True)
                    NodeEditor.set_active_node(self, False)
                imgui.close_current_popup()
            _duplicate, _ = imgui.menu_item("Duplicate node")
            if _duplicate:
                duplicate_node = Node.create_node_by_type(self.type)
                duplicate_node.position = [self.position[0] + 10, self.position[1] + 10]
                NodeEditor.set_active_node(duplicate_node)
            _delete, _ = imgui.menu_item("Delete node")
            if _delete:
                self.delete()
            _reset, _ = imgui.menu_item("Reset node")
            if _reset:
                new_node = Node.create_node_by_type(self.type)
                new_node.position = self.position
                self.delete()
            imgui.end_popup()
        if NodeEditor.profiling and self.does_profiling_time:
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
        window_position = (self.last_measured_window_position[0] + self.size[0] - Node.PLAY_BUTTON_SIZE / 2, self.last_measured_window_position[1] + self.size[1] / 2 - Node.PLAY_BUTTON_SIZE / 2)

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
                                 imgui.get_color_u32_rgba(*Node.colour_whiten(Node.COLOUR[self.type])))
        drawlist.add_rect_filled(Node.PROGRESS_BAR_PADDING + origin[0], y,
                                 Node.PROGRESS_BAR_PADDING + origin[0] + width * min([1.0, progress]),
                                 y + Node.PROGRESS_BAR_HEIGHT, imgui.get_color_u32_rgba(*Node.COLOUR[self.type]))

    def delete(self):
        print(NodeEditor.nodes)
        try:
            if NodeEditor.focused_node == self:
                NodeEditor.set_active_node(None, True)
            if NodeEditor.active_node == self:
                NodeEditor.set_active_node(None)
            NodeEditor.nodes.remove(self)
            for attribute in self.connectable_attributes:
                attribute.delete()
        except Exception as e:
            NodeEditor.set_error(e, "Problem deleting node\n"+str(e))
        print(NodeEditor.nodes)

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
        elif node_type == Node.TYPE_PARTICLE_FILTER:
            return ParticleFilterNode()
        elif node_type == Node.TYPE_BAKE_STACK:
            return BakeStackNode()
        elif node_type == Node.TYPE_CROP_IMAGE:
            return CropImageNode()
        else:
            return False

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

        if node.type == Node.TYPE_LOAD_DATA:
            return node
        try:
            source = get_any_incoming_node(node)
            while not isinstance(source, NullNode):
                load_data_node = source.type == Node.TYPE_LOAD_DATA
                if load_data_node:
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
        if NodeEditor.profiling:
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
            NodeEditor.set_error(e, f"{Node.TITLE[self.type]} error: "+str(e))
        if NodeEditor.profiling:
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
        pass

    def pre_save(self):
        NodeEditor.pickle_temp = dict()
        self.last_index_requested = -1
        self.last_frame_returned = None
        NodeEditor.pickle_temp["profiler_time"] = self.profiler_time
        self.pre_save_impl()

    def pre_save_impl(self):
        pass

    def post_save(self):
        self.profiler_time = NodeEditor.pickle_temp["profiler_time"]
        self.post_save_impl()
        NodeEditor.pickle_temp = dict()

    def post_save_impl(self):
        pass


class NullNode(Node):
    def __init__(self):
        super().__init__(Node.TYPE_NULL)
        self.dataset = Dataset()
        self.dataset_in = None
        self.pixel_size = 100

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
        if NodeEditor.active_connector == self:
            if NodeEditor.active_connector_hover_pos is not None:
                draw_list = imgui.get_background_draw_list()
                draw_list.add_line(self.draw_x, self.draw_y, *NodeEditor.active_connector_hover_pos, imgui.get_color_u32_rgba(*ConnectableAttribute.CONNECTION_LINE_COLOUR), ConnectableAttribute.CONNECTION_LINE_THICKNESS)
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
            if NodeEditor.connector_delete_requested:
                NodeEditor.connector_delete_requested = False
                self.disconnect_all()
                self.any_change = True

        ## Allow draw source if currently no node being edited
        if NodeEditor.active_connector is self or NodeEditor.active_connector is None:
            if imgui.begin_drag_drop_source(imgui.DRAG_DROP_SOURCE_NO_PREVIEW_TOOLTIP):
                NodeEditor.active_connector = self
                NodeEditor.active_connector_parent_node = self.parent
                imgui.set_drag_drop_payload('connector', b'1')  # Arbitrary payload value - manually keeping track of the payload elsewhere
                imgui.end_drag_drop_source()
        elif NodeEditor.active_connector is not None:
            imgui.begin_child(f"Attribute{self.id}drop_target")
            imgui.end_child()
            if imgui.begin_drag_drop_target():
                if NodeEditor.connector_released:
                    self.connect_attributes(NodeEditor.active_connector)
                    NodeEditor.connector_released = False
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
        NodeEditor.active_connector_parent_node = None

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


class LoadDataNode(Node):

    def __init__(self):
        super().__init__(Node.TYPE_LOAD_DATA) #Was: super(LoadDataNode, self).__init__()
        self.size = [200, 200]

        # Set up connectable attributes
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, parent=self)
        self.connectable_attributes.append(self.dataset_out)
        # Set up node-specific vars
        self.dataset = Dataset()
        self.path = ""
        self.pixel_size = 64.0
        self.load_on_the_fly = True
        self.done_loading = False
        self.to_load_idx = 0
        self.n_to_load = 1

        self.file_filter_positive_raw = ""
        self.file_filter_negative_raw = ""

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_out.render_end()
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
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
            imgui.push_item_width(45)
            _c, self.pixel_size = imgui.input_float("##nm", self.pixel_size, 0.0, 0.0, format = "%.1f")
            self.any_change = self.any_change or _c
            if _c:
                self.dataset.pixel_size = self.pixel_size
            imgui.pop_item_width()
            imgui.same_line()
            imgui.text("nm")
            imgui.columns(1)


            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()

    def render_advanced(self):
        _, self.load_on_the_fly = imgui.checkbox("Load on the fly", self.load_on_the_fly)
        if not self.load_on_the_fly and not self.done_loading:
            self.progress_bar(min([self.to_load_idx / self.n_to_load]))
            imgui.spacing()
            imgui.spacing()
            imgui.spacing()

        imgui.text("Title must contain:")
        av_width = imgui.get_content_region_available_width()
        imgui.set_next_item_width(av_width)
        _c, self.file_filter_positive_raw = imgui.input_text("##filt_pos", self.file_filter_positive_raw, 1024, imgui.INPUT_TEXT_AUTO_SELECT_ALL)
        Node.tooltip("Enter tags, e.g. 'GFP;RFP', that must be in a filename\n"
                     "in order to retain the frame. Separate tags by a semicolon.\n"
                     "When multiple tags are entered, frames are retained if \n"
                     "any of the tags is in the filename (not necessarily all).\n"
                     "Leave empty to retain all files by default.")
        imgui.text("Title must not contain:")
        imgui.set_next_item_width(av_width)
        _c, self.file_filter_negative_raw = imgui.input_text("##filt_neg", self.file_filter_negative_raw, 1024, imgui.INPUT_TEXT_AUTO_SELECT_ALL)
        Node.tooltip("Enter tags, e.g. 'GFP;RFP', to select frames for deletion.\n"
                     "Separate by a semicolon. When the positive and negative\n"
                     "selection criteria contradict, frames are retained.")
        if imgui.button("Filter", av_width / 2 - 5, 25):
            self.dataset.filter_frames_by_title(self.file_filter_positive_raw, self.file_filter_negative_raw)
            self.any_change = True
        imgui.same_line(spacing=10)
        if imgui.button("Reset", av_width / 2 - 5, 25):
            self.on_select_file()
            self.any_change = True

    def on_select_file(self):
        self.dataset = Dataset(self.path, self.pixel_size)
        self.n_to_load = self.dataset.n_frames
        self.done_loading = False
        self.to_load_idx = 0
        self.any_change = True
        cfg.image_viewer.center_image_requested = True
        NodeEditor.set_active_node(self)

    def get_image_impl(self, idx):
        if self.dataset.n_frames > 0:
            retimg = copy.deepcopy(self.dataset.get_indexed_image(idx))
            retimg.clean()
            return retimg
        else:
            return None

    def on_update(self):
        if not self.load_on_the_fly and not self.done_loading:
            if NodeEditor.profiling:
                time_start = time.time()
                self.profiler_count += 1
            if self.to_load_idx < self.dataset.n_frames:
                self.dataset.get_indexed_image(self.to_load_idx).load()
                self.to_load_idx += 1
            else:
                self.done_loading = True
            if NodeEditor.profiling:
                self.profiler_time += time.time() - time_start

    def pre_save_impl(self):
        NodeEditor.pickle_temp["dataset"] = self.dataset
        self.dataset = Dataset()

    def post_save_impl(self):
        self.dataset = NodeEditor.pickle_temp["dataset"]


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
        self.buffer_last_output = True
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
            imgui.push_item_width(140)
            _method_changed, self.register_method = imgui.combo("Method", self.register_method, RegisterNode.METHODS)
            _reference_changed, self.reference_method = imgui.combo("Reference", self.reference_method, RegisterNode.REFERENCES)
            imgui.pop_item_width()
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
            if self.any_change:
                self.reference_image = None
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            input_img = data_source.get_image(idx)
            if self.reference_image is None:
                # Get reference frame according to specified pairing method
                if self.reference_method == 2:
                    self.reference_image = data_source.get_image(idx - 1).load()
                elif self.reference_method == 0:
                    self.reference_image = self.image_in.get_incoming_node().get_image(idx=None).load()
                elif self.reference_method == 1:
                    self.reference_image = data_source.get_image(self.frame).load()

            # Perform registration according to specified registration method
            if self.register_method == 0:
                if self.reference_image is not None:
                    template = self.reference_image
                    image = input_img.load()
                    if self.use_roi:
                        template = template[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                        image = image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                    tmat = self.sr.register(template, image)
                    input_img.translation = [tmat[0][2], tmat[1][2]]
            else:
                NodeEditor.set_error(Exception(), "RegisterNode: reference method not available (may not be implemented yet).")

            if self.reference_method == 2:
                self.reference_image = None
            input_img.bake_transform()
            return input_img

    def pre_save_impl(self):
        self.reference_image = None


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
        self.pixel_size = 1

        self.roi = [0, 0, 0, 0]

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
                if NodeEditor.profiling:
                    self.profiler_count += 1
                if self.mode == 0:
                    image_in = datasource.get_image(self.frame)
                    self.image = image_in.load()
                    self.pixel_size = image_in.pixel_size
                elif self.mode == 1:
                    load_data_node = Node.get_source_load_data_node(self)
                    self.pixel_size = load_data_node.dataset.pixel_size
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
        if NodeEditor.profiling:
            self.profiler_count -= 1
        if self.any_change:
            self.configure_settings()
        if self.image is not None:
            out_frame = Frame("virtual_frame")
            out_frame.data = self.image
            out_frame.pixel_size = self.pixel_size
            return out_frame


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
            if self.filter == 0:
                chosen_wavelet = SpatialFilterNode.WAVELETS[SpatialFilterNode.WAVELET_NAMES[self.wavelet]]
                if chosen_wavelet is None:
                    chosen_wavelet = self.custom_wavelet
                in_pxd = input_image.load()
                input_image.data= pywt.swt2(in_pxd, wavelet=chosen_wavelet, level=self.level, norm=True, trim_approx=True)[0]
            elif self.filter == 1:
                input_image.data = gaussian_filter(input_image.load(), self.sigma)
            elif self.filter == 2:
                input_image.data = medfilt(input_image.load(), self.kernel)
            return input_image
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

            out_image = data_source.get_image(idx)
            out_image.data = pxd
            return out_image


class FrameShiftNode(Node):
    def __init__(self):
        super().__init__(Node.TYPE_FRAME_SHIFT)
        self.size = [140, 100]

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

        self.magnification = 10
        self.default_sigma = 30.0
        self.fix_sigma = False

        self.reconstructor = Reconstructor()
        self.latest_image = None

        self.original_pixel_size = 100.0
        self.reconstruction_pixel_size = 10.0
        self.reconstruction_image_size = [1, 1]
        self.paint_particles = False
        self.paint_currently_applied = False

        paint_in = ConnectableAttribute(ConnectableAttribute.TYPE_COLOUR, ConnectableAttribute.INPUT, parent=self)
        self.connectable_attributes.append(paint_in)
        self.particle_painters = [paint_in]
        self.does_profiling_time = False
        self.does_profiling_count = False

    def render(self):
        if super().render_start():
            self.reconstruction_in.render_start()
            self.image_out.render_start()
            self.reconstruction_in.render_end()
            self.image_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(100)
            _mag_changed, self.magnification = imgui.input_int("Magnification", self.magnification, 1, 1)
            self.magnification = max([self.magnification, 1])
            imgui.text(f"Final pixel size: {self.original_pixel_size / self.magnification:.1f}")
            imgui.text(f"Final image size: {self.reconstruction_image_size[0]} x {self.reconstruction_image_size[1]} px")

            _c, self.fix_sigma = imgui.checkbox("Force uncertainty", self.fix_sigma)


            if self.fix_sigma:
                imgui.same_line(spacing=10)
                imgui.push_item_width(50)
                _, self.default_sigma = imgui.input_float(" nm", self.default_sigma, 0, 0, format="%.1f")
                imgui.pop_item_width()

            # Colourize options
            _c, self.paint_particles = imgui.checkbox("Paint particles", self.paint_particles)
            if _c and not self.paint_particles:
                for i in range(len(self.particle_painters) - 1):
                    self.particle_painters[i].delete()

            imgui.spacing()
            if self.paint_particles:
                for connector in self.particle_painters:
                    # add / remove slots
                    if connector.check_connect_event():
                        new_slot = ConnectableAttribute(ConnectableAttribute.TYPE_COLOUR, ConnectableAttribute.INPUT, parent=self)
                        self.particle_painters.append(new_slot)
                        self.connectable_attributes.append(new_slot)
                    elif connector.check_disconnect_event():
                        self.particle_painters.remove(connector)
                        self.connectable_attributes.remove(connector)

                    # render blobs
                    imgui.new_line()
                    connector.render_start()
                    connector.render_end()

            if imgui.button("Render", 50, 20):
                self.build_reconstruction()


            if _mag_changed:
                self.original_pixel_size = Node.get_source_load_data_node(self).dataset.pixel_size
                roi = self.get_particle_data().reconstruction_roi
                img_width = int((roi[2] - roi[0]) * self.magnification)
                img_height = int((roi[3] - roi[1]) * self.magnification)
                self.reconstruction_image_size = (img_width, img_height)

            super().render_end()

    def update_pixel_size(self):
        image_width = int((self.get_particle_data().reconstruction_roi[2] - self.get_particle_data().reconstruction_roi[0]) * self.magnification)
        image_height = int((self.get_particle_data().reconstruction_roi[3] - self.get_particle_data().reconstruction_roi[1]) * self.magnification)
        self.reconstruction_image_size = (image_width, image_height)
        new_pixel_size = self.original_pixel_size / self.magnification
        if new_pixel_size != self.reconstruction_pixel_size:
            self.reconstruction_pixel_size = new_pixel_size
            self.reconstructor.set_pixel_size(self.reconstruction_pixel_size)

    def build_reconstruction(self):
        try:
            self.original_pixel_size = Node.get_source_load_data_node(self).dataset.pixel_size
            self.update_pixel_size()
            self.reconstructor.set_pixel_size(self.original_pixel_size / self.magnification)
            self.reconstructor.set_image_size(self.reconstruction_image_size)
            datasource = self.reconstruction_in.get_incoming_node()
            if datasource:
                particle_data = datasource.get_particle_data()
                self.reconstructor.set_particle_data(particle_data)
                self.reconstructor.set_camera_origin([-particle_data.reconstruction_roi[0] * self.magnification, -particle_data.reconstruction_roi[1] * self.magnification])

                ## Apply colours
                if self.paint_particles:
                    if len(self.particle_painters) > 0:
                        for particle in particle_data.particles:
                            particle.colour = np.asarray([0.0, 0.0, 0.0])
                    for i in range(0, len(self.particle_painters) - 1):
                        self.particle_painters[i].get_incoming_node().apply_paint_to_particledata(particle_data)
                    self.paint_currently_applied = True
                else:
                    if self.paint_currently_applied:
                        for particle in particle_data.particles:
                            particle.colour = np.asarray([1.0, 1.0, 1.0])
                    self.paint_currently_applied = False

                if self.reconstructor.particle_data.empty:
                    return None
                else:
                    self.latest_image = self.reconstructor.render(fixed_uncertainty=(self.default_sigma if self.fix_sigma else None))
                    self.any_change = True
            else:
                self.latest_image = None
        except Exception as e:
            NodeEditor.set_error(e, "Error building reconstruction.\n"+str(e))

    def get_image_impl(self, idx=None):
        if self.latest_image is not None:
            img_wrapper = Frame("super-resolution reconstruction virtual frame")
            img_wrapper.data = self.latest_image
            img_wrapper.pixel_size = self.reconstruction_pixel_size
            return img_wrapper
        else:
            return None

    def get_particle_data_impl(self):
        datasource = self.reconstruction_in.get_incoming_node()
        if datasource:
            return datasource.get_particle_data()
        else:
            return ParticleData()

    def on_gain_focus(self):
        self.original_pixel_size = Node.get_source_load_data_node(self).dataset.pixel_size
        roi = self.get_particle_data().reconstruction_roi
        img_width = int((roi[2] - roi[0]) * self.magnification)
        img_height = int((roi[3] - roi[1]) * self.magnification)
        self.reconstruction_image_size = (img_width, img_height)

    def pre_save_impl(self):
        NodeEditor.pickle_temp["latest_image"] = self.latest_image
        self.latest_image = None

    def post_save_impl(self):
        self.latest_image = NodeEditor.pickle_temp["latest_image"]


class ParticlePainterNode(Node):

    def __init__(self):
        super().__init__(Node.TYPE_PARTICLE_PAINTER)  # Was: super(LoadDataNode, self).__init__()
        self.size = [270, 240]

        self.reconstruction_in = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.INPUT, parent = self)
        self.colour_out = ConnectableAttribute(ConnectableAttribute.TYPE_COLOUR, ConnectableAttribute.OUTPUT, parent = self)

        self.connectable_attributes.append(self.reconstruction_in)
        self.connectable_attributes.append(self.colour_out)

        self.parameter = 0
        self.available_parameters = ["Fixed colour"]
        self.colour_min = (0.0, 0.0, 0.0)
        self.colour_max = (1.0, 1.0, 1.0)
        self.histogram_values = np.asarray([0, 0]).astype('float32')
        self.histogram_bins = np.asarray([0, 0]).astype('float32')
        self.histogram_initiated = False
        self.min = 0
        self.max = 0
        self.paint_dry = True  # flag to notify whether settings were changed that would result in different particle colours.
        self.returns_image = False
        self.does_profiling_count = False

    def render(self):
        if super().render_start():
            self.reconstruction_in.render_start()
            self.colour_out.render_start()
            self.reconstruction_in.render_end()
            self.colour_out.render_end()

            if self.reconstruction_in.check_connect_event() or self.reconstruction_in.check_disconnect_event():
                self.init_histogram_values()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            imgui.push_item_width(160)
            _c, self.parameter = imgui.combo("Parameter", self.parameter, self.available_parameters)
            self.paint_dry = self.paint_dry or _c
            if _c:
                self.get_histogram_values()
                self.min = self.histogram_bins[0]
                self.max = self.histogram_bins[-1]
            imgui.pop_item_width()

            if self.parameter == 0:
                imgui.same_line()
                _c, self.colour_min = imgui.color_edit3("##Colour_min", *self.colour_min, imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL)
                self.colour_max = self.colour_max
                self.paint_dry = self.paint_dry or _c
            else:
                _c, self.colour_min = imgui.color_edit3("##Colour_min", *self.colour_min, imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL)
                self.paint_dry = self.paint_dry or _c
                imgui.same_line(spacing = 20)
                imgui.text("min")
                imgui.same_line(spacing=imgui.get_content_region_available_width() - 35 - imgui.get_font_size() * len("max") * 2)
                imgui.text("max")
                imgui.same_line(spacing = 20)
                _c, self.colour_max = imgui.color_edit3("##Colour_max", *self.colour_max, imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL | imgui.COLOR_EDIT_NO_SIDE_PREVIEW)
                self.paint_dry = self.paint_dry or _c

                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *self.colour_max)
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, *self.colour_max)
                content_width = imgui.get_content_region_available_width()
                imgui.plot_histogram("##hist", self.histogram_values, graph_size = (content_width, 40))
                imgui.text("{:.2f}".format(self.histogram_bins[0]))
                imgui.same_line(position = content_width - imgui.get_font_size() * len(str(self.histogram_bins[-1])) / 2)
                imgui.text("{:.2f}".format(self.histogram_bins[-1]))
                imgui.push_item_width(content_width)
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *self.colour_min)
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *self.colour_min)
                _c, self.min = imgui.slider_float("##min", self.min, self.histogram_bins[0], self.histogram_bins[-1], format = "min: %1.2f")
                self.paint_dry = self.paint_dry or _c
                imgui.pop_style_color(2)
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *self.colour_max)
                imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *self.colour_max)
                _c, self.max = imgui.slider_float("##max", self.max, self.histogram_bins[0], self.histogram_bins[-1], format="max: %1.2f")
                self.paint_dry = self.paint_dry or _c
                imgui.pop_style_color(4)
                imgui.pop_item_width()

            super().render_end()

    def get_histogram_values(self):
        datasource = self.reconstruction_in.get_incoming_node()
        if datasource:
            particledata = datasource.get_particle_data()
            if self.available_parameters[self.parameter] in list(particledata.histogram_counts.keys()):
                self.histogram_values = particledata.histogram_counts[self.available_parameters[self.parameter]]
                self.histogram_bins = particledata.histogram_bins[self.available_parameters[self.parameter]]

    def init_histogram_values(self):
        datasource = self.reconstruction_in.get_incoming_node()

        if datasource:
            self.histogram_initiated = True
            particledata = datasource.get_particle_data()
            if particledata.baked:
                self.available_parameters = ["Fixed colour"] + list(particledata.histogram_counts.keys())
                self.parameter = min([1, len(self.available_parameters)])
                self.histogram_values = particledata.histogram_counts[self.available_parameters[self.parameter]]
                self.histogram_bins = particledata.histogram_bins[self.available_parameters[self.parameter]]
                self.min = self.histogram_bins[0]
                self.max = self.histogram_bins[-1]

    def apply_paint_to_particledata(self, particledata):
        if NodeEditor.profiling:
            time_start = time.time()
        if self.parameter == 0:
            _colour = np.asarray(self.colour_max)
            for particle in particledata.particles:
                particle.colour += _colour
        else:
            i = 0
            c_min = np.asarray(self.colour_min)
            c_max = np.asarray(self.colour_max)
            values = particledata.parameter[self.available_parameters[self.parameter]]
            for particle in particledata.particles:
                fac = min([max([(values[i] - self.min) / self.max, 0.0]), 1.0])
                _colour = (1.0 - fac) * c_min + fac * c_max
                particle.colour += _colour
                i += 1
        if NodeEditor.profiling:
            self.profiler_time += time.time() - time_start

    def on_gain_focus(self):
        if self.histogram_initiated:
            self.get_histogram_values()


class ParticleDetectionNode(Node):
    METHODS = ["Local maximum"]
    THRESHOLD_OPTIONS = ["Value", "St. Dev.", "Mean", "Max", "Min"]

    def __init__(self):
        super().__init__(Node.TYPE_PARTICLE_DETECTION)
        self.size = [290, 205]

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.localizations_out = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.OUTPUT, parent=self)

        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.localizations_out)

        self.method = 0
        self.thresholding = 1
        self.threshold = 100
        self.sigmas = 2.0
        self.means = 3.0
        self.n_max = 2500
        self.d_min = 1
        self.max_fac = 0.75
        self.min_fac = 5.0
        self.roi = [0, 0, 0, 0]

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.localizations_out.render_start()
            self.dataset_in.render_end()
            self.localizations_out.render_end()

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
                _c, self.threshold = imgui.input_int("Value", self.threshold, 0, 0)
                self.any_change = self.any_change or _c
            elif self.thresholding == 1:
                _c, self.sigmas = imgui.slider_float("x Sigma", self.sigmas, 1.0, 5.0, format = "%.1f")
                self.any_change = self.any_change or _c
            elif self.thresholding == 2:
                _c, self.means = imgui.slider_float("x Mean", self.means, 0.1, 10.0, format = "%.1f")
                self.any_change = self.any_change or _c
            elif self.thresholding == 3:
                _c, self.max_fac = imgui.slider_float("x Max", self.max_fac, 0.0, 1.0, format="%.1f")
                self.any_change = self.any_change or _c
            elif self.thresholding == 4:
                _c, self.min_fac = imgui.slider_float("x Min", self.min_fac, 1.0, 10.0, format="%.1f")
                self.any_change = self.any_change or _c
            Node.tooltip("The final threshold value is determined by this factor x the metric (sigma, mean, etc.)\n"
                         "chosen above. In case of Threshold method 'Value', the value entered here is the final \n"
                         "threshold value. Note that the metric is calculated for the ROI only.")
            _c, self.n_max = imgui.slider_int("Max particles", self.n_max, 100, 2500)
            Node.tooltip("Maximum amount of particles output per frame. Ctrl + click to override the slider\n"
                         "and enter a custom value.")
            self.any_change = self.any_change or _c

            imgui.pop_item_width()
            imgui.push_item_width(100)
            _c, self.d_min = imgui.input_int("Minimum distance (px)", self.d_min, 1, 10)
            self.any_change = self.any_change or _c
            self.d_min = max([1, self.d_min])
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        source = self.dataset_in.get_incoming_node()
        if NodeEditor.profiling and self.frame_requested_by_image_viewer:
            self.profiler_count += 1
        if source is not None:
            # Find threshold value
            image_obj = source.get_image(idx)
            image = image_obj.load()
            if self.use_roi:
                image = image[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
            threshold = self.threshold
            if self.thresholding == 1:
                threshold = self.sigmas * np.std(image)
            elif self.thresholding == 2:
                threshold = self.means * np.mean(image)
            elif self.thresholding == 3:
                threshold = self.max_fac * np.amax(image)
            elif self.thresholding == 4:
                threshold = self.min_fac * np.amin(image)
            # Perform requested detection method
            coordinates = peak_local_max(image, threshold_abs = threshold, num_peaks = self.n_max, min_distance = self.d_min)
            if self.use_roi:
                coordinates += np.asarray([self.roi[1], self.roi[0]])
                Node.get_source_load_data_node(self).dataset.reconstruction_roi = self.roi
            else:
                Node.get_source_load_data_node(self).dataset.reconstruction_roi = [0, 0, image.shape[0], image.shape[1]]
            image_obj.maxima = coordinates

            return image_obj

    def get_coordinates(self, idx=None):
        try:
            if NodeEditor.profiling:
                time_start = time.time()
                self.profiler_count += 1
            retval = self.get_image_impl(idx).maxima
            if NodeEditor.profiling:
                self.profiler_time += time.time() - time_start
            return retval
        except Exception as e:
            NodeEditor.set_error(e, "Error returning coordinates "+str(e))


class ExportDataNode(Node):

    def __init__(self):
        super().__init__(Node.TYPE_EXPORT_DATA)
        self.size = [210, 220]

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_MULTI, ConnectableAttribute.INPUT, self, [ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.TYPE_IMAGE, ConnectableAttribute.TYPE_RECONSTRUCTION])
        self.dataset_in.title = "Any"
        self.connectable_attributes.append(self.dataset_in)

        self.path = "..."
        self.roi = [0, 0, 0, 0]
        self.include_discarded_frames = False
        self.saving = False
        self.export_type = 0  # 0 for dataset, 1 for image.
        self.frames_to_load = list()
        self.n_frames_to_save = 1
        self.n_frames_saved = 0

        self.returns_image = False
        self.does_profiling_count = False

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.dataset_in.render_end()

            self.export_type = -1
            if self.dataset_in.get_incoming_attribute_type() == ConnectableAttribute.TYPE_DATASET:
                self.export_type = 0
            elif self.dataset_in.get_incoming_attribute_type() == ConnectableAttribute.TYPE_IMAGE:
                self.export_type = 1
            elif self.dataset_in.get_incoming_attribute_type() == ConnectableAttribute.TYPE_RECONSTRUCTION:
                self.export_type = 2
            else:
                self.dataset_in.title = "Any"
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if self.export_type in [0, 1]:
                _, self.use_roi = imgui.checkbox("use ROI", self.use_roi)
            imgui.text("Output path")
            imgui.push_item_width(150)
            _, self.path = imgui.input_text("##intxt", self.path, 256, imgui.INPUT_TEXT_ALWAYS_OVERWRITE)
            imgui.pop_item_width()
            imgui.same_line()
            if imgui.button("...", 26, 19):
                filename = filedialog.asksaveasfilename()
                if filename is not None:
                    if '.' in filename[-5:]:
                        filename = filename[:filename.rfind(".")]
                    self.path = filename

            content_width = imgui.get_window_width()
            save_button_width = 100
            save_button_height = 40

            if self.saving:
                imgui.spacing()
                imgui.text("Export progress")
                self.progress_bar(self.n_frames_saved / self.n_frames_to_save)

            imgui.spacing()
            imgui.spacing()
            imgui.spacing()
            imgui.new_line()
            imgui.same_line(position=(content_width - save_button_width) // 2)
            if not self.saving:
                if imgui.button("Save", save_button_width, save_button_height):
                    self.do_save()
            else:
                if imgui.button("Cancel", save_button_width, save_button_height):
                    self.saving = False
            super().render_end()

    def get_img_and_save(self, idx):
        img_pxd = self.get_image_impl(idx)
        if self.use_roi:
            img_pxd = img_pxd[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
        Image.fromarray(img_pxd).save(self.path+"/0"+str(idx)+".tif")

    def do_save(self):
        if NodeEditor.profiling:
            time_start = time.time()
        if self.export_type == 0:  # Save stack
            self.saving = True
            #if self.include_discarded_frames:
                #n_active_frames = Node.get_source_load_data_node(self).dataset.n_frames
                #self.frames_to_load = list(range(0, n_active_frames))
            self.frames_to_load = list()
            for i in range(Node.get_source_load_data_node(self).dataset.n_frames):
                self.frames_to_load.append(i)
            self.n_frames_to_save = len(self.frames_to_load)
            self.n_frames_saved = 0
            if not os.path.isdir(self.path):
                os.mkdir(self.path)
        elif self.export_type == 1:  # Save image
            img = self.dataset_in.get_incoming_node().get_image(idx=None)
            img_pxd = img.load()
            if self.use_roi:
                img_pxd = img_pxd[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
            try:
                util.save_tiff(img_pxd, self.path+".tif", pixel_size_nm=img.pixel_size)
            except Exception as e:
                NodeEditor.set_error(e, "Error saving image: "+str(e))
        elif self.export_type == 2: # Save particle data
            try:
                self.dataset_in.get_incoming_node().get_particle_data().save_as_csv(self.path+".csv")
            except Exception as e:
                NodeEditor.set_error(e, "Error saving .csv\n"+str(e))
        if NodeEditor.profiling:
            self.profiler_time += time.time() - time_start

    def on_update(self):
        if self.saving:
            try:
                indices = list()
                for i in range(min([NodeEditor.batch_size, len(self.frames_to_load)])):
                    self.n_frames_saved += 1
                    indices.append(self.frames_to_load[-1])
                    self.frames_to_load.pop()
                Parallel(n_jobs=NodeEditor.batch_size)(delayed(self.get_img_and_save)(index) for index in indices)
                if len(self.frames_to_load) == 0:
                    self.saving = False
            except Exception as e:
                self.saving = False
                NodeEditor.set_error(e, "Error saving stack: \n"+str(e))

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            incoming_img = data_source.get_image(idx)
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

            imgui.push_item_width(100)
            _c, self.mode = imgui.combo("Method", self.mode, BinImageNode.MODES)
            self.any_change = _c or self.any_change
            imgui.pop_item_width()
            imgui.push_item_width(60)
            _c, self.factor = imgui.input_int("Bin factor", self.factor, 0, 0)
            self.any_change = _c or self.any_change
            self.factor = max([self.factor, 1])
            imgui.pop_item_width()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            image_in = data_source.get_image(idx)
            pxd = image_in.load()
            width, height = pxd.shape
            pxd = pxd[:self.factor * (width // self.factor), :self.factor * (height // self.factor)]
            if self.mode == 0:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).mean(2).mean(0)
            elif self.mode == 1:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).median(2).median(0)
            elif self.mode == 2:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).min(2).min(0)
            elif self.mode == 3:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).max(2).max(0)
            elif self.mode == 4:
                pxd = pxd.reshape((self.factor, width // self.factor, self.factor, height // self.factor)).sum(2).sum(0)
            image_in.data = pxd
            return image_in


class ParticleFittingNode(Node):
    RANGE_OPTIONS = ["All frames", "Current frame only", "Custom range"]
    ESTIMATORS = ["Least squares (GPU)", "Maximum likelihood (GPU)", "No estimator (CPU, no parallel)"]
    PSFS = ["Gaussian", "Elliptical Gaussian"]

    def __init__(self):
        super().__init__(Node.TYPE_PARTICLE_FITTING)

        # Set up connectable attributes
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.localizations_in = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.INPUT, parent=self)
        self.reconstruction_out = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.OUTPUT, parent=self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.reconstruction_out)
        self.connectable_attributes.append(self.localizations_in)

        self.size = [300, 270]
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
        self.particle_data = ParticleData()

        self.time_start = 0
        self.time_stop = 0

        self.intensity_min = 100.0
        self.intensity_max = -1.0
        self.sigma_min = 1.0
        self.sigma_max = 10.0
        self.offset_min = 0.0
        self.offset_max = -1.0

        self.custom_bounds = False

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.reconstruction_out.render_start()
            self.dataset_in.render_end()
            self.reconstruction_out.render_end()
            imgui.new_line()
            self.localizations_in.render_start()
            self.localizations_in.render_end()

            imgui.spacing()
            imgui.separator()
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
            _c, self.range_option = imgui.combo("Range", self.range_option,
                                                ParticleFittingNode.RANGE_OPTIONS)
            self.any_change = _c or self.any_change
            if self.range_option == 2:
                imgui.push_item_width(80)
                _c, (self.range_min, self.range_max) = imgui.input_int2('[start, stop) index', self.range_min,
                                                                        self.range_max)
                self.any_change = _c or self.any_change
                imgui.pop_item_width()
            imgui.spacing()
            if self.fitting:
                self.progress_bar(self.n_fitted / self.n_to_fit)
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()

            _play_btn_clicked, _state = self.play_button()
            if _play_btn_clicked and _state:
                self.init_fit()
            elif _play_btn_clicked and self.fitting:
                self.fitting = False

            ## TODO: add camera offset and ADU per photon (to LOAD DATASET?)
            if not self.particle_data.empty:
                imgui.text("Reconstruction info")
                imgui.text(f"particles: {len(self.particle_data.particles)}")
                if not self.fitting:
                    imgui.text(f"x range: {self.particle_data.x_min / 1000.0:.1f} to {self.particle_data.x_max / 1000.0:.1f} um")
                    imgui.text(f"y range: {self.particle_data.y_min / 1000.0:.1f} to {self.particle_data.y_max / 1000.0:.1f} um")
            imgui.spacing()
            header_expanded, _ = imgui.collapsing_header("Advanced", None)
            if header_expanded:
                self.render_advanced()

            super().render_end()

    def render_advanced(self):
        _c, self.custom_bounds = imgui.checkbox("Use custom parameter bounds", self.custom_bounds)
        Node.tooltip("Edit the bounds for particle parameters intensity, sigma, and offset.")
        if self.custom_bounds:
            imgui.push_item_width(45)
            _c, self.sigma_min = imgui.input_float("min##sigma", self.sigma_min, 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle sigma to a minimum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            _c, self.sigma_max = imgui.input_float("max##sigma", self.sigma_max, 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle sigma to a maximum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            imgui.text("sigma (px)")

            _c, self.intensity_min = imgui.input_float("min##int", self.intensity_min, 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle intensity to a minimum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            _c, self.intensity_max = imgui.input_float("max##int", self.intensity_max, 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle intensity to a maximum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            imgui.text("intensity (counts)")

            _c, self.offset_min = imgui.input_float("min##offset", self.offset_min, 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle offset to a minimum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            _c, self.offset_max = imgui.input_float("max##offset", self.offset_max, 0, 0, "%.1f")
            Node.tooltip("Enter value to limit particle offset to a maximum value, or '-1.0' to leave unbounded.")
            imgui.same_line()
            imgui.text("offset (counts)")

            imgui.pop_item_width()

    def init_fit(self):
        try:
            self.time_start = time.time()
            dataset_source = Node.get_source_load_data_node(self)
            self.particle_data = ParticleData(dataset_source.pixel_size)
            self.particle_data.set_reconstruction_roi(dataset_source.dataset.reconstruction_roi)
            self.fitting = True
            self.frames_to_fit = list()
            if self.range_option == 0:
                dataset = dataset_source.dataset
                n_frames = dataset.n_frames
                self.frames_to_fit = list(range(0, n_frames))
            elif self.range_option == 2:
                self.frames_to_fit = list(range(self.range_min, self.range_max))
            elif self.range_option == 1:
                dataset = dataset_source.dataset
                self.frames_to_fit = [dataset.current_frame]
            self.n_to_fit = len(self.frames_to_fit)
            self.n_fitted = 0
        except Exception as e:
            self.fitting = False
            NodeEditor.set_error(e, "Error in init_fit: "+str(e))

    def on_update(self):
        try:
            if self.fitting:
                if len(self.frames_to_fit) == 0:
                    self.fitting = False
                    self.play = False
                    self.particle_data.bake()
                else:
                    fitted_frame = self.get_image(self.frames_to_fit[-1])
                    self.n_fitted += 1
                    if fitted_frame is not None:
                        particles = fitted_frame.particles
                        self.frames_to_fit.pop()
                        if not fitted_frame.discard:
                            self.particle_data += particles
        except Exception as e:
            self.fitting = False
            self.play = False
            NodeEditor.set_error(e, "Error in ParticleFitNode.on_update(self): "+str(e))

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        coord_source = self.localizations_in.get_incoming_node()
        if data_source and coord_source:
            frame = data_source.get_image(idx)
            coordinates = coord_source.get_coordinates(idx)
            frame.maxima = coordinates
            if frame.discard:
                frame.particles = None
                return frame
            particles = list()
            if self.estimator in [0, 1]:
                particles = pfit.frame_to_particles(frame, self.initial_sigma, self.estimator, self.crop_radius, constraints = [self.intensity_min, self.intensity_max, -1, -1, -1, -1, self.sigma_min, self.sigma_max, self.offset_min, self.offset_max])
            elif self.estimator == 2:
                particles = list()
                pxd = frame.load()
                for i in range(len(frame.maxima)):
                    intensity = pxd[frame.maxima[i, 0], frame.maxima[i, 1]]
                    x = frame.maxima[i, 0]
                    y = frame.maxima[i, 1]
                    particles.append(Particle(idx, x, y, 1.0, intensity))
            frame.particles = particles
            new_maxima = list()
            for particle in frame.particles:
                new_maxima.append([particle.y, particle.x])
            frame.maxima = new_maxima
            return frame

    def get_particle_data_impl(self):
        self.particle_data.clean()
        return self.particle_data

    def pre_save_impl(self):
        NodeEditor.pickle_temp["particle_data"] = self.particle_data
        self.particle_data = ParticleData()

    def post_save_impl(self):
        self.particle_data = NodeEditor.pickle_temp["particle_data"]


class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__(Node.TYPE_PARTICLE_FILTER)

        self.reconstruction_in = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.INPUT, parent=self)
        self.reconstruction_out = ConnectableAttribute(ConnectableAttribute.TYPE_RECONSTRUCTION, ConnectableAttribute.OUTPUT, parent=self)

        self.connectable_attributes.append(self.reconstruction_in)
        self.connectable_attributes.append(self.reconstruction_out)

        self.size = [310, 250]
        self.available_parameters = ["No data available"]
        self.filters = list()

        self.returns_image = False
        self.does_profiling_count = False

    def render(self):
        if super().render_start():
            self.reconstruction_in.render_start()
            self.reconstruction_out.render_start()
            self.reconstruction_in.render_end()
            self.reconstruction_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            if self.reconstruction_in.check_connect_event():
                self.get_histogram_parameters()

            i = 0
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, 0.2, 0.2, 0.2, 1.0)
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM_HOVERED, 0.2, 0.2, 0.2, 1.0)
            for pf in self.filters:
                imgui.push_id(f"pfid{i}")
                if imgui.button("-", 20, 20):
                    self.filters.remove(pf)
                    imgui.pop_id()
                    continue
                imgui.same_line()
                imgui.set_next_item_width(144)
                _c, pf.parameter = imgui.combo("Criterion", pf.parameter, self.available_parameters)
                if _c:
                    pf.set_data(*self.get_histogram_vals(self.available_parameters[pf.parameter]), self.available_parameters[pf.parameter])
                imgui.same_line(spacing = 10)
                _c, pf.invert = imgui.checkbox("NOT", pf.invert)
                Node.tooltip("If NOT is selected, the filter logic is inverted. Default\n"
                             "behaviour is: particle retained if min < parameter < max.\n"
                             "With NOT on, a particle is retained if max < parameter OR\nmin > parameter")

                pf.render()
                imgui.pop_id()
                i += 1
            content_width = imgui.get_window_content_region_width()
            imgui.new_line()
            imgui.same_line(content_width / 2 - 50)
            if imgui.button("Add filter", width = 100, height = 20):
                new_filter = ParticleFilterNode.Filter(self)
                new_filter.set_data(*self.get_histogram_vals(self.available_parameters[0]), self.available_parameters[0])
                self.filters.append(new_filter)
            imgui.pop_style_color(2)
            super().render_end()

    def on_gain_focus(self):
        self.get_histogram_parameters()

    def get_histogram_vals(self, parameter):
        datasource = self.reconstruction_in.get_incoming_node()
        if datasource:
            particledata = datasource.get_particle_data()
            return particledata.histogram_counts[parameter], particledata.histogram_bins[parameter]
        else:
            return np.asarray([0, 0]).astype('float32'), np.asarray([0, 1]).astype('float32')

    def get_histogram_parameters(self):
        datasource = self.reconstruction_in.get_incoming_node()
        if datasource:
            particledata = datasource.get_particle_data()
            self.available_parameters = list(particledata.histogram_counts.keys())

    def get_particle_data_impl(self):
        if NodeEditor.profiling:
            time_start = time.time()
        datasource = self.reconstruction_in.get_incoming_node()
        if datasource:
            pdata = datasource.get_particle_data()
            for pf in self.filters:
                pf.apply(pdata)
            if NodeEditor.profiling:
                self.profiler_time += time.time() - time_start
            return pdata
        else:
            if NodeEditor.profiling:
                self.profiler_time += time.time() - time_start
            return ParticleData()

    class Filter:
        def __init__(self, parent):
            self.parameter_key = ""
            self.vals = [0, 0]
            self.bins = [0, 0]
            self.min = 0
            self.max = 1
            self.parameter = 0
            self.invert = False
            self.parent = parent

        def set_data(self, vals, bins, prm_key):
            self.vals = vals
            self.bins = bins
            self.min = bins[0]
            self.max = bins[-1]
            self.parameter_key = prm_key

        def render(self):
            content_width = imgui.get_content_region_available_width()
            imgui.plot_histogram("##histogram", self.vals, graph_size = (content_width, 40))
            imgui.text("{:.2f}".format(self.bins[0]))
            imgui.same_line(position=content_width - imgui.get_font_size() * len(str(self.bins[-1])) / 2)
            imgui.text("{:.2f}".format(self.bins[-1]))
            imgui.push_item_width(content_width)
            _c, self.min = imgui.slider_float("##min", self.min, self.bins[0], self.bins[-1], "min: %1.2f")
            _c, self.max = imgui.slider_float("##max", self.max, self.bins[0], self.bins[-1], "max: %1.2f")
            imgui.pop_item_width()
            imgui.separator()
            imgui.spacing()

        def apply(self, particle_data_object):
            if NodeEditor.profiling:
                time_start = time.time()
            particle_data_object.apply_filter(self.parameter_key, self.min, self.max, logic_not=self.invert)
            if NodeEditor.profiling:
                self.parent.profiler_time += time.time() - time_start


class BakeStackNode(Node):
    RANGE_OPTIONS = ["All frames", "Custom range"]

    def __init__(self):
        super().__init__(Node.TYPE_BAKE_STACK)# make it take dataset AND coordinates.
        self.size = [230, 100]

        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT, self)
        self.coordinates_in = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.INPUT, self)
        self.coordinates_out = ConnectableAttribute(ConnectableAttribute.TYPE_COORDINATES, ConnectableAttribute.OUTPUT, self)

        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)
        self.connectable_attributes.append(self.coordinates_in)
        self.connectable_attributes.append(self.coordinates_out)

        self.parallel = True
        self.baking = False
        self.n_to_bake = 1
        self.n_baked = 0
        self.range_option = 0
        self.custom_range_min = 0
        self.custom_range_max = 1
        self.has_dataset = False
        self.dataset = Dataset()
        self.frames_to_bake = list()
        self.temp_dir_do_not_edit = "_srnodes_temp"+str(self.id)
        self.baked_at = 0
        self.has_coordinates = False
        self.coordinates = list()
        self.bake_coordinates = False

    def render(self):
        if super().render_start():
            self.dataset_in.render_start()
            self.dataset_in.render_end()
            if self.has_dataset:
                self.dataset_out.render_start()
                self.dataset_out.render_end()
            if self.bake_coordinates:
                imgui.spacing()
                self.coordinates_in.render_start()
                self.coordinates_in.render_end()
                if self.has_coordinates:
                    self.coordinates_out.render_start()
                    self.coordinates_out.render_end()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            _c, self.bake_coordinates = imgui.checkbox("Also bake coordinates", self.bake_coordinates)
            if _c and not self.bake_coordinates:
                self.coordinates_in.disconnect_all()

            if self.range_option == 1:
                imgui.push_item_width(80)
                _c, (self.custom_range_min, self.custom_range_max) = imgui.input_int2('[start, top) index', self.custom_range_min, self.custom_range_max)
                imgui.pop_item_width()
            _c, self.parallel = imgui.checkbox("Parallel processing", self.parallel)
            imgui.set_next_item_width(100)
            _c, self.range_option = imgui.combo("Range to bake", self.range_option, BakeStackNode.RANGE_OPTIONS)
            clicked, self.baking = self.play_button()
            if clicked and self.baking:
                self.init_bake()
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()

            if self.baking:
                imgui.text("Baking process:")
                self.progress_bar(self.n_baked / self.n_to_bake)
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()
            if self.has_dataset:
                imgui.text(f"Output data was baked at: {self.baked_at}")
            super().render_end()

    def get_image_and_save(self, idx=None):
        datasource = self.dataset_in.get_incoming_node()
        if datasource:
            pxd = datasource.get_image(idx).load()
            Image.fromarray(pxd).save(self.temp_dir_do_not_edit + "/0" + str(idx) + ".tif")
            if self.bake_coordinates:
                coordsource = self.coordinates_in.get_incoming_node()
                coordinates = coordsource.get_coordinates(idx)
                return coordinates
            return None
        return None

    def get_image_impl(self, idx=None):
        if self.has_dataset and idx in range(0, self.dataset.n_frames):
            retimg = copy.deepcopy(self.dataset.get_indexed_image(idx))
            retimg.clean()
            if self.has_coordinates:
                retimg.maxima = self.coordinates[idx]
            return retimg
        else:
            datasource = self.dataset_in.get_incoming_node()
            if datasource:
                return datasource.get_image(idx)

    def get_coordinates(self, idx=None):
        if self.has_coordinates and idx in range(0, self.dataset.n_frames):
            return self.coordinates[idx]

    def init_bake(self):

        self.baked_at = datetime.datetime.now().strftime("%H:%M")
        if os.path.isdir(self.temp_dir_do_not_edit):
            shutil.rmtree(self.temp_dir_do_not_edit)
        os.mkdir(self.temp_dir_do_not_edit)
        self.has_dataset = False
        self.has_coordinates = False
        self.coordinates = list()
        if self.range_option == 0:
            dataset_source = Node.get_source_load_data_node(self)
            self.frames_to_bake = list(range(0, dataset_source.dataset.n_frames))
        elif self.range_option == 1:
            self.frames_to_bake = list(range(self.custom_range_min, self.custom_range_max))
        self.n_to_bake = max([1, len(self.frames_to_bake)])

    def on_update(self):
        if self.baking:
            if NodeEditor.profiling:
                time_start = time.time()
            try:
                indices = list()
                for i in range(min([NodeEditor.batch_size, len(self.frames_to_bake)])):
                    self.n_baked += 1
                    indices.append(self.frames_to_bake[-1])
                    self.frames_to_bake.pop()
                coordinates = Parallel(n_jobs=NodeEditor.batch_size)(delayed(self.get_image_and_save)(index) for index in indices)
                if self.bake_coordinates:
                    self.coordinates += coordinates
                if NodeEditor.profiling:
                    self.profiler_time += time.time() - time_start
                    self.profiler_count += len(indices)

                if len(self.frames_to_bake) == 0:
                    # Load dataset from temp dir.
                    self.dataset = Dataset(self.temp_dir_do_not_edit + "/00.tif")
                    print("Done baking - now loading from disk")
                    for frame in self.dataset.frames:
                        frame.load()
                    print("Done loading")
                    self.any_change = True
                    self.baking = False
                    self.play = False
                    self.has_dataset = True
                    if self.bake_coordinates:
                        self.coordinates.reverse()
                        self.has_coordinates = True
            except Exception as e:
                self.baking = False
                self.play = False
                NodeEditor.set_error(e, "Error baking stack: \n"+str(e))


class CropImageNode(Node):
    def __init__(self):
        super().__init__(Node.TYPE_CROP_IMAGE)
        self.size = [140, 100]
        self.dataset_in = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.INPUT, parent=self)
        self.dataset_out = ConnectableAttribute(ConnectableAttribute.TYPE_DATASET, ConnectableAttribute.OUTPUT,parent=self)
        self.connectable_attributes.append(self.dataset_in)
        self.connectable_attributes.append(self.dataset_out)

        self.roi = [0, 0, 0, 0]
        self.use_roi = True

    def render(self):
        if super().render_start():
            self.dataset_out.render_start()
            self.dataset_in.render_start()
            self.dataset_out.render_end()
            self.dataset_in.render_end()
            super().render_end()

    def get_image_impl(self, idx=None):
        data_source = self.dataset_in.get_incoming_node()
        if data_source:
            if self.frame_requested_by_image_viewer:
                return data_source.get_image(idx)
            else:
                out_frame = data_source.get_image(idx).clone()
                pxd = out_frame.load()[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
                out_frame.data = pxd
                out_frame.width, out_frame.height = out_frame.data.shape
                return out_frame
