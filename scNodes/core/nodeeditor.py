import importlib.util
import shutil
import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
import sys
import tkinter as tk
from tkinter import filedialog
from scNodes.core.node import *
import os
import pyperclip
import dill as pickle
from scNodes.core.opengl_classes import Texture
import json
tkroot = tk.Tk()
tkroot.withdraw()


class NodeEditor:
    TOOLTIP_APPEAR_DELAY = 1.0  # seconds
    TOOLTIP_HOVERED_TIMER = 0.0
    TOOLTIP_HOVERED_START_TIME = 0.0

    NODE_FACTORY = dict()
    NODE_GROUPS = dict()

    def __init__(self, window, imgui_context, imgui_impl):
        self.window = window
        self.window.clear_color = cfg.COLOUR_WINDOW_BACKGROUND
        self.window.make_current()

        self.imgui_context = imgui_context
        self.imgui_implementation = imgui_impl


        # Context menu
        self.context_menu_position = [0, 0]
        self.context_menu_open = False
        self.context_menu_can_close = False

        NodeEditor.init_node_factory()

        if True:
            self.boot_img_texture = Texture(format="rgba32f")
            pxd_boot_img_texture = np.asarray(Image.open(cfg.root+"icons/scnodes_boot_img.png")).astype(np.float32) / 255.0
            self.boot_img_texture.update(pxd_boot_img_texture)
            self.boot_img_height, self.boot_img_width = pxd_boot_img_texture.shape[0:2]
            self.boot_img_texture.set_linear_interpolation()
            self.show_boot_img = True

    def get_font_atlas_ptr(self):
        return self.imgui_implementation.io.fonts

    def on_update(self):
        if cfg.node_editor_relink:
            NodeEditor.relink_after_load()
            cfg.node_editor_relink = False
        if cfg.next_active_node is not None:
            cfg.active_node = cfg.next_active_node
            cfg.next_active_node = None

        imgui.set_current_context(self.imgui_context)
        self.window.make_current()
        self.window.set_full_viewport()
        if self.window.focused:
            self.imgui_implementation.process_inputs()
        self.window.on_update()
        if self.window.window_size_changed:
            cfg.window_width = self.window.width
            cfg.window_height = self.window.height
        if not imgui.get_io().want_capture_keyboard and imgui.is_key_pressed(glfw.KEY_TAB):
            cfg.active_editor = 1
        if not self.window.get_key(glfw.KEY_ESCAPE):
            for node in cfg.nodes:
                node.clear_flags()
                node.on_update()
        else:
            pass
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
        if self.window.get_mouse_event(glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS, pop_event=False):
            cfg.connector_delete_requested = True

        cfg.node_move_requested = [0, 0]
        cfg.camera_move_requested = [0, 0]
        imgui_want_mouse = imgui.get_io().want_capture_mouse

        if self.window.get_mouse_button(glfw.MOUSE_BUTTON_LEFT) and imgui_want_mouse:
            cfg.node_move_requested = self.window.cursor_delta
        elif self.window.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE) and not imgui_want_mouse:
            cfg.camera_move_requested = self.window.cursor_delta

        imgui.get_io().display_size = self.window.width, self.window.height
        imgui.new_frame()
        self._gui_main()
        imgui.render()
        self.imgui_implementation.render(imgui.get_draw_data())

    def end_frame(self):
        self.window.end_frame()

    def _gui_main(self):
        # Render nodes  - render active_connector_parent_node first, to enable all other connectors' drop targets.
        source_node_id = -1
        if cfg.active_connector_parent_node is not None:
            source_node_id = cfg.active_connector_parent_node.id
            cfg.active_connector_parent_node.render()
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

        NodeEditor._menu_bar()
        self._warning_window()
        self._boot_img()

    def _boot_img(self):
        if self.show_boot_img:
            imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *cfg.COLOUR_WINDOW_BACKGROUND[0:3], 0.0)
            imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *cfg.COLOUR_WINDOW_BACKGROUND[0:3], 0.0)
            imgui.push_style_color(imgui.COLOR_TEXT, *(0.0, 0.0, 0.0, 1.0))

            _w = self.boot_img_width * 0.5
            _h = self.boot_img_height * 0.5
            imgui.set_next_window_position((cfg.window_width - _w) / 2.0, (cfg.window_height - _h) / 2.0 - 25)
            self.show_boot_img = imgui.begin("##bootwindow", True, imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_BACKGROUND)[1]
            imgui.image(self.boot_img_texture.renderer_id, _w, _h)
            imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *cfg.COLOUR_WINDOW_BACKGROUND)
            if imgui.begin_popup_context_window():
                imgui.text(f"version {cfg.version}\nsource: github.com/bionanopatterning/scNodes")
                imgui.end_popup()
            if self.window.focused and imgui.is_mouse_clicked(glfw.MOUSE_BUTTON_LEFT) and not imgui.is_window_hovered():
                self.show_boot_img = False
            imgui.pop_style_color(1)
            imgui.end()
            imgui.pop_style_color(3)

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
            _, stay_open = imgui.begin("Notifications", True, imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE)
            if not cfg.error_logged:
                cfg.write_to_log(cfg.error_msg)
                cfg.error_logged = True
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

    def _context_menu(self):
        imgui.set_next_window_position(self.context_menu_position[0] - 3, self.context_menu_position[1] - 3)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *cfg.COLOUR_MENU_WINDOW_BACKGROUND)
        imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_CM_WINDOW_TEXT)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *cfg.COLOUR_MENU_WINDOW_BACKGROUND)
        imgui.begin("##necontextmenu", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        # Close context menu when it is not hovered.
        context_menu_hovered = imgui.is_window_hovered(flags=imgui.HOVERED_ALLOW_WHEN_BLOCKED_BY_POPUP | imgui.HOVERED_CHILD_WINDOWS)
        if context_menu_hovered:
            self.context_menu_can_close = True
        if not context_menu_hovered and self.context_menu_can_close:
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

    @staticmethod
    def _menu_bar():
        imgui.push_style_color(imgui.COLOR_MENUBAR_BACKGROUND, *cfg.COLOUR_MAIN_MENU_BAR)
        imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_MAIN_MENU_BAR_TEXT)
        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *cfg.COLOUR_MAIN_MENU_BAR_HILIGHT)
        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *cfg.COLOUR_MAIN_MENU_BAR_HILIGHT)
        imgui.push_style_color(imgui.COLOR_HEADER, *cfg.COLOUR_MAIN_MENU_BAR_HILIGHT)
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *cfg.COLOUR_MENU_WINDOW_BACKGROUND)
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (2.0, 2.0))
        ## Save node setup.
        if imgui.core.begin_main_menu_bar():
            if imgui.begin_menu('File'):
                save_setup, _ = imgui.menu_item("Save setup")
                if save_setup:
                    filename = filedialog.asksaveasfilename(filetypes=[("scNodes setup", ".scn")])
                    if filename != '':
                        NodeEditor.save_node_setup(filename)
                load_setup, _ = imgui.menu_item("Load setup")
                if load_setup:
                    try:
                        filename = filedialog.askopenfilename(filetypes=[("scNodes setup", ".scn")])
                        if filename != '':
                            cfg.nodes = list()
                            NodeEditor.append_node_setup(filename)
                    except Exception as e:
                        cfg.set_error(e, f"Error loading node setup:\n"+str(e))
                import_setup, _ = imgui.menu_item("Append setup")
                if import_setup:
                    try:
                        filename = filedialog.askopenfilename(filetypes=[("scNodes setup", ".scn")])
                        if filename != '':
                            NodeEditor.append_node_setup(filename)
                    except Exception as e:
                        cfg.set_error(e, "Error importing node setup:\n"+str(e))
                clear_setup, _ = imgui.menu_item("Clear setup")
                if clear_setup:
                    for i in range(len(cfg.nodes)):
                        cfg.nodes[0].delete()
                    cfg.nodes = list()
                imgui.end_menu()
            if imgui.begin_menu('Settings'):
                install_node, _ = imgui.menu_item("Install a node")
                if install_node:
                    NodeEditor.install_node()
                if imgui.begin_menu('Profiling'):
                    _c, cfg.profiling = imgui.checkbox("Track node processing times", cfg.profiling)
                    NodeEditor.tooltip("Keep track of the time that every node in the pipeline takes to process and output a frame.\n")
                    clear_times, _ = imgui.menu_item("   Reset timers")
                    if clear_times:
                        for node in cfg.nodes:
                            node.profiler_time = 0.0
                            node.profiler_count = 0
                    imgui.end_menu()
                if imgui.begin_menu('Parallel processing'):
                    imgui.set_next_item_width(26)
                    _c, cfg.batch_size = imgui.input_int("batch size", cfg.batch_size, 0, 0)
                    NodeEditor.tooltip(
                        "Number of frames to process within one parallel processing batch. Values higher than the amount\n"
                        "of CPUs on the PC are allowed and will result in multiple tasks being dispatched to individual\n"
                        "CPUs per batch. This increases processing speed, but reduces GUI responsiveness. For optimal\n"
                        "efficiency, set the batch size to an integer multiple of the amount of CPUs on the machine. \n"
                        f"This PC has: {cfg.n_cpus_max} CPUs.\n")
                    imgui.end_menu()
                imgui.end_menu()
            if imgui.begin_menu('Editor'):
                select_node_editor, _ = imgui.menu_item("Node Editor", None, selected=True)
                select_correlation_editor, _ = imgui.menu_item("Correlation", None, selected=False)
                if select_correlation_editor:
                    cfg.active_editor = 1
                imgui.end_menu()
            imgui.end_main_menu_bar()
        imgui.pop_style_color(6)
        imgui.pop_style_var(1)

    @staticmethod
    def install_node():
        try:
            filename = filedialog.askopenfilename(filetypes=[("Python file", ".py")])
            if filename != '':
                node_dir = __file__[:__file__.rfind("\\")]+"/nodes/"
                node_dir = node_dir.replace('\\', '/')
                node_name = filename[filename.rfind("/")+1:]
                shutil.copyfile(filename, node_dir+node_name)
            cfg.set_error(Exception(), "Node installed! Restarting the software is required for it to become available.\n\n\n")
        except Exception as e:
            cfg.set_error(e, "Error upon installing node. Are you sure you selected the right file?")

    @staticmethod
    def init_node_factory():
        nodeimpls = list()

        class NodeImpl:
            def __init__(self, node_create_fn):
                self.create_fn = node_create_fn
                self.node_obj = node_create_fn()
                self.title = self.node_obj.title
                self.group = self.node_obj.group
                self.id = self.node_obj.sortid
                self.enabled = self.node_obj.enabled

            def __del__(self):
                self.node_obj.delete()

        node_source_files = glob.glob(cfg.root+"nodes/*.py")  # load all .py files in the /nodes folder
        i = 0
        for nodesrc in node_source_files:  # for every file, dynamically load module and save the module's create() function to a dict, keyed by name of node.
            i += 1
            if "__init__.py" in nodesrc:
                continue

            module_name = nodesrc[nodesrc.rfind("\\")+1:-3]
            try:
                # get the module spec and import the module
                mod = importlib.import_module(("scNodes." if not cfg.frozen else "")+"nodes."+module_name)
                impl = NodeImpl(mod.create)
                if not impl.enabled:
                    continue
                nodeimpls.append(impl)

            except Exception as e:
                cfg.nodes = list()
                cfg.set_error(e, f"No well-defined Node type found in {nodesrc}. See manual for minimal code requirements.")
        node_ids = list()

        for ni in nodeimpls:
            node_ids.append(ni.id)
        sorted_id_indices = np.argsort(node_ids)

        for nid in sorted_id_indices:
            _node = nodeimpls[nid]
            NodeEditor.NODE_FACTORY[_node.title] = _node.create_fn
            if isinstance(_node.group, str):
                if _node.group not in NodeEditor.NODE_GROUPS.keys():
                    NodeEditor.NODE_GROUPS[_node.group] = list()
                NodeEditor.NODE_GROUPS[_node.group].append(_node.title)
            elif isinstance(_node.group, list):
                for group in _node.group:
                    if group not in NodeEditor.NODE_GROUPS.keys():
                        NodeEditor.NODE_GROUPS[group] = list()
                    NodeEditor.NODE_GROUPS[group].append(_node.title)

        NodeEditor.node_group_all = list(NodeEditor.NODE_FACTORY.keys())

    @staticmethod
    def save_node_setup(path):
        if path[-len(cfg.filetype_node_setup):] != cfg.filetype_node_setup:
            path+=cfg.filetype_node_setup
        try:
            node_setup = list()
            for node in cfg.nodes:
                node_setup.append(node.to_dict())
            links = list()
            for node in cfg.nodes:
                for home_attribute in node.connectable_attributes.values():
                    if home_attribute.direction == ConnectableAttribute.INPUT:
                        continue
                    for partner_attribute in home_attribute.linked_attributes:
                        links.append([home_attribute.id, partner_attribute.id])

            full_setup = dict()
            full_setup["nodes"] = node_setup
            full_setup["links"] = links
            print(full_setup)
            print(node_setup)
            print(links)
            with open(path, 'w') as outfile:
                json.dump(full_setup, outfile, indent=2)
        except Exception as e:
            cfg.set_error(e, "Error saving node setup")


    @staticmethod
    def append_node_setup(path):
        try:
            with open(path, 'r') as infile:
                full_setup = json.load(infile)
                # parse the node setup
                node_setup = full_setup["nodes"]
                new_nodes = list()
                for node_dict in node_setup:
                    # generate a node of the right type:
                    title = node_dict["title"]
                    new_node = NodeEditor.NODE_FACTORY[title]()
                    new_node.from_dict(node_dict)
                    new_nodes.append(new_node)
                # with all the nodes and their attributes added, link up the attributes:
                all_attributes_by_id = dict()
                for node in cfg.nodes:
                    for a in node.connectable_attributes.values():
                        if a.id in node.connectable_attributes:
                            cfg.set_error(BaseException(), "Error appending node setup - an id was used multiple times.")
                        all_attributes_by_id[a.id] = a
                links = full_setup["links"]
                for link in links:
                    home_attribute = all_attributes_by_id[link[0]]
                    parent_attribute = all_attributes_by_id[link[1]]
                    home_attribute.connect_attributes(parent_attribute)
                # finally, call all of the new nodes' on_load function
                for node in new_nodes:
                    node.on_load()
        except Exception as e:
            cfg.set_error(e, "Error appending/loading node setup: ")


    @staticmethod
    def relink_after_load():
        nodes = cfg.nodes
        attributes = list()
        ids_to_link = list()
        for node in nodes:
            for attribute in node.connectable_attributes:
                attributes.append(attribute)
                if attribute.direction == ConnectableAttribute.INPUT:
                    for partner in attribute.linked_attributes:
                        ids_to_link.append((attribute.id, partner.id))
                #attribute.disconnect_all()

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
        dirs = glob.glob("_SRNTEMP*/")
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

