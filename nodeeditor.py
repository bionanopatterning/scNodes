import importlib.util
import shutil
import imgui
from imgui.integrations.glfw import GlfwRenderer
import time
import glfw
import glob
import dill as pickle
import copy
import sys
import config as cfg
import tkinter as tk
from tkinter import filedialog
tkroot = tk.Tk()
tkroot.withdraw()


class NodeEditor:
    COLOUR_WINDOW_BACKGROUND = (0.94, 0.94, 0.94, 1.0)
    CONTEXT_MENU_SIZE = (200, 98)
    ERROR_WINDOW_HEIGHT = 80
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
    TOOLTIP_APPEAR_DELAY = 1.0  # seconds
    TOOLTIP_HOVERED_TIMER = 0.0
    TOOLTIP_HOVERED_START_TIME = 0.0

    NODE_FACTORY = dict()
    NODE_GROUPS = dict()

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

        NodeEditor.init_node_factory()

    def get_font_atlas_ptr(self):
        return self.imgui_implementation.io.fonts

    def on_update(self):
        if cfg.next_active_node is not None:
            cfg.active_node = cfg.next_active_node
            cfg.next_active_node = None

        imgui.set_current_context(self.imgui_context)
        self.window.make_current()
        self.window.set_full_viewport()
        if self.window.focused:
            self.imgui_implementation.process_inputs()
        self.window.on_update()
        cfg.window_width = self.window.width
        cfg.window_height = self.window.height

        if not self.window.get_key(glfw.KEY_ESCAPE):
            for node in cfg.nodes:
                node.clear_flags()
                node.on_update()
        else:
            print("Esc pressed - skipping on_update for all nodes.")
        if cfg.focused_node is not None:
            cfg.focused_node.any_change = cfg.any_change
        cfg.any_change = False
        cfg.connector_delete_requested = False
        cfg.active_connector_hover_pos = self.window.cursor_pos
        if cfg.active_connector is not None:
            if not self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT):
                cfg.connector_released = True
                cfg.active_connector_hover_pos = None
                cfg.active_connector = None
            else:
                cfg.connector_released = False
        if self.window.get_mouse_event(glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, pop_event = False):
            cfg.connector_delete_requested = True

        cfg.node_move_requested = [0, 0]
        cfg.camera_move_requested = [0, 0]
        imgui_want_mouse = imgui.get_io().want_capture_mouse
        if self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT) and imgui_want_mouse:
            cfg.node_move_requested = self.window.cursor_delta
        elif self.window.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE) and not imgui_want_mouse:
            cfg.camera_move_requested = self.window.cursor_delta

        imgui.new_frame()
        self._gui_main()
        imgui.render()
        self.imgui_implementation.render(imgui.get_draw_data())

    def end_frame(self):
        self.window.end_frame()

    def _gui_main(self):

        ## Render nodes  - render active_connector_parent_node first, to enable all other connectors' drop targets.
        source_node_id = -1
        if cfg.active_connector_parent_node is not None:
            cfg.active_connector_parent_node.render()
            source_node_id = cfg.active_connector_parent_node.id
        for node in cfg.nodes:
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
        if cfg.error_msg is not None:
            if cfg.error_new:
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
            imgui.text(cfg.error_msg)
            if imgui.button("(debug): raise error", 180, 20):
                raise cfg.error_obj
            if imgui.is_window_focused() and self.window.get_mouse_event(glfw.MOUSE_BUTTON_LEFT, glfw.PRESS):
                cfg.error_new = False
            imgui.end()
            if not stay_open:
                cfg.error_msg = None
                cfg.error_new = True
            imgui.pop_style_color(5)
            imgui.pop_style_var(1)

    def _context_menu(self):
        imgui.set_next_window_position(self.context_menu_position[0] - 3, self.context_menu_position[1] - 3)
        imgui.set_next_window_size_constraints((NodeEditor.CONTEXT_MENU_SIZE[0], 10), (NodeEditor.CONTEXT_MENU_SIZE[0], 500))
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *NodeEditor.COLOUR_MENU_WINDOW_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_TEXT, *NodeEditor.COLOUR_CM_WINDOW_TEXT)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *NodeEditor.COLOUR_MENU_WINDOW_BACKGROUND)
        imgui.begin("##necontextmenu", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        # Close context menu when it is not hovered.
        context_menu_hovered = imgui.is_window_hovered(flags=imgui.HOVERED_ALLOW_WHEN_BLOCKED_BY_POPUP | imgui.HOVERED_CHILD_WINDOWS)
        if context_menu_hovered:
            self.context_menu_can_close = True
        if not context_menu_hovered and self.context_menu_can_close:
            self.context_menu_can_close = False
            self.context_menu_open = False

        # Context menu contents
        new_node = None
        for key in NodeEditor.NODE_GROUPS:
            if imgui.begin_menu(key):
                for nodetitle in NodeEditor.NODE_GROUPS[key]:
                    item_selected, _ = imgui.menu_item(nodetitle)
                    if item_selected:
                        new_node = NodeEditor.NODE_FACTORY[nodetitle]()
                imgui.end_menu()
        if new_node:
            try:
                new_node.position = self.context_menu_position
                cfg.set_active_node(new_node)
                self.context_menu_open = False
            except Exception as e:
                cfg.set_error(e, "Error upon requesting new node (probably not implemented yet.)\nError:"+str(e))

        if cfg.focused_node is not None:
            clear_active_node, _ = imgui.menu_item("Release focused node")
            if clear_active_node:
                cfg.set_active_node(None, True)
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
                                cfg.nodes = imported_nodes
                                NodeEditor.relink_after_load()
                    except Exception as e:
                        cfg.set_error(e, f"Error loading node setup - are you sure you selected a '.srn' file?\n"+str(e))
                import_setup, _ = imgui.menu_item("Append node setup")
                if import_setup:
                    try:
                        filename = filedialog.askopenfilename(filetypes=[("srNodes setup", ".srn")])
                        if filename != '':
                            with open(filename, 'rb') as pickle_file:
                                imported_nodes = pickle.load(pickle_file)
                                for node in imported_nodes:
                                    cfg.nodes.append(node)
                                NodeEditor.relink_after_load()
                    except Exception as e:
                        cfg.set_error(e, "Error importing node setup - are you sure you selected a '.srn' file?\n"+str(e))
                save_setup, _ = imgui.menu_item("Save current setup")
                if save_setup:
                    filename = filedialog.asksaveasfilename(filetypes=[("srNodes setup", ".srn")])
                    if filename != '':
                        if filename[-4:] == ".srn":
                            filename = filename[:-4]
                        with open(filename+".srn", 'wb') as pickle_file:
                            node_copies = list()
                            for node in cfg.nodes:
                                node.pre_save()
                                node_copies.append(copy.deepcopy(node))
                                node.post_save()
                            pickle.dump(node_copies, pickle_file)
                clear_setup, _ = imgui.menu_item("Clear current setup")
                if clear_setup:
                    for i in range(len(cfg.nodes)):
                        cfg.nodes[0].delete()
                    cfg.nodes = list()
                imgui.end_menu()
            if imgui.begin_menu('Settings'):
                if imgui.begin_menu('Profiling'):
                    _c, cfg.profiling = imgui.checkbox("Track node processing times", cfg.profiling)
                    NodeEditor.tooltip("Keep track of the time that every node in the pipeline takes to process and output a frame.\n")
                    clear_times, _ = imgui.menu_item("   Reset timers")
                    if clear_times:
                        for node in cfg.nodes:
                            node.profiler_time = 0.0
                            node.profiler_count = 0
                    imgui.end_menu()
                imgui.set_next_item_width(30)
                _c, cfg.n_cpus = imgui.input_int("Parallel batch size", cfg.n_cpus, 0, 0)
                if _c:
                    cfg.n_cpus = cfg.n_cpus
                    cfg.batch_size = cfg.n_cpus

                NodeEditor.tooltip("Number of frames to process within one parallel processing batch. Values higher than the amount\n" 
                             "of CPUs on the PC are allowed and will result in multiple tasks being dispatched to individual\n" 
                             "CPUs per batch. This can increase processing speed, but reduces GUI responsiveness. For optimal\n" 
                             "efficiency, set the batch size to an integer multiple of the amount of CPUs on the machine. \n"
                             f"This PC has: {cfg.n_cpus_max} CPUs.\n"
                             f"Set to '-1' to force use of all CPUs.")
                imgui.end_menu()
            imgui.end_main_menu_bar()
        imgui.pop_style_color(5)

    @staticmethod
    def init_node_factory():
        node_source_files = glob.glob("nodes/*.py")
        i = 0
        for nodesrc in node_source_files:
            if "custom_node_template" in nodesrc:
                continue
            # give the module a unique name
            _name = f"dynamicnodemodule{i}"

            try:
                # get the module spec and import the module
                spec = importlib.util.spec_from_file_location(_name, nodesrc)
                temp = importlib.util.module_from_spec(spec)
                sys.modules[_name] = temp
                spec.loader.exec_module(temp)

                # add the module and its create method to the nodefactory
                _node = temp.create()
                NodeEditor.NODE_FACTORY[_node.title] = temp.create
                print(_node.group)
                if isinstance(_node.group, str):
                    if _node.group not in NodeEditor.NODE_GROUPS.keys():
                        NodeEditor.NODE_GROUPS[_node.group] = list()
                    NodeEditor.NODE_GROUPS[_node.group].append(_node.title)
                elif isinstance(_node.group, list):
                    for group in _node.group:
                        if group not in NodeEditor.NODE_GROUPS.keys():
                            NodeEditor.NODE_GROUPS[group] = list()
                        NodeEditor.NODE_GROUPS[group].append(_node.title)
                _node.delete()

            except Exception as e:
                print(f"NodeEditor - init node factory didn't work for source file\n{nodesrc}")
                raise e
        NodeEditor.node_group_all = list(NodeEditor.NODE_FACTORY.keys())

    @staticmethod
    def relink_after_load():
        nodes = cfg.nodes
        attributes = list()
        ids_to_link = list()
        for node in nodes:
            for attribute in node.connectable_attributes:
                attributes.append(attribute)
                if attribute.direction == True:  # TODO: replace == True with something verbose like == ConnectableAttribute.INPUT (which it was before)
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

    @staticmethod
    def tooltip(text):
        if imgui.is_item_hovered():
            if NodeEditor.TOOLTIP_HOVERED_TIMER == 0.0:
                NodeEditor.TOOLTIP_HOVERED_START_TIME = time.time()
                NodeEditor.TOOLTIP_HOVERED_TIMER = 0.001  # add a fake 1 ms to get out of this if clause
            elif NodeEditor.TOOLTIP_HOVERED_TIMER > NodeEditor.TOOLTIP_APPEAR_DELAY:
                imgui.set_tooltip(text)
            else:
                NodeEditor.TOOLTIP_HOVERED_TIMER = time.time() - NodeEditor.TOOLTIP_HOVERED_START_TIME
        if not imgui.is_any_item_hovered():
            NodeEditor.TOOLTIP_HOVERED_TIMER = 0.0

